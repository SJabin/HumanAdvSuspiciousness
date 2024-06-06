import argparse
#import logging
import time
import os
import random
from math import ceil
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import bert_params, roberta_params, xlnet_params, bart_params, albert_params, distilbert_params, sbert_params
from utils import convert_examples_to_features, IMDBDataset, MovieReviewDataset, MnliDataset, TrustPilotDataset
import torch
from torch.nn import  MSELoss, L1Loss, PoissonNLLLoss, KLDivLoss, HuberLoss

from sklearn.metrics import mean_squared_log_error,r2_score,mean_squared_error
#from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
#from sklearn.ensemble import VotingRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.linear_model import ElasticNet
#from sklearn.linear_model import Ridge,LinearRegression

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW, RMSprop, SGD
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression,HuberRegressor
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel, BartTokenizer, BartModel
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr, spearmanr
#from sklearn.preprocessing import MinMaxScaler
from torch import optim
from model import Regressor, build_model, convert_examples_to_features

from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    set_seed
)
os.environ["WANDB_DISABLED"] = "true"
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback
from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular

from SMART.smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss, convert_examples_to_features, build_model


#Losses and optimizers
MSE_par = {'name': 'mean-squared-error', 'loss_fc':MSELoss}
L1_par = {'name': 'mean-absolute-error', 'loss_fc':L1Loss}
Poisson_par = {'name': 'Poisson', 'loss_fc':PoissonNLLLoss}
KL_par = {'name': 'Kulback-Leiblar-divergence', 'loss_fc':KLDivLoss}
Huber_par = {'name': 'Huber', 'loss_fc':HuberLoss}
AdamW_par = {'name':'AdamW', 'fc':AdamW}
RMS_par = {'name':'RMS', 'fc':RMSprop}
SGD_par = {'name':'SGD', 'fc':SGD}


#metrices
def compute_accuracy(preds, labels):
    #preds_flat = np.argmax(preds, axis=1).flatten()
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    return np.sum(preds_flat == labels_flat) / len(labels)

def compute_pearson(predicts, labels):
    res = pearsonr(labels, predicts)
    pcof = res[0]
    pval = res[1]
    return pcof, pval #*100

def compute_spearman(predicts, labels):
    res = spearmanr(labels, predicts)
    scof = res[0]
    pval = res[1]
    return scof, pval

def compute_rmse(predicts, labels):
    rmse=np.sqrt(mean_squared_error(labels,predicts))
    return rmse

def calc_classification_metrics(p: EvalPrediction):
    #p.predictions - [loss, logits ,layer_outs ]
    predictions = p.predictions[0].flatten()
#     print(predictions)
#     pred_labels = np.argmax(predictions, axis=1)
#     pred_scores = softmax(predictions, axis=1)[:, 1]
    labels = p.label_ids
    
    #acc = compute_accuracy(predictions, labels)
    pearson, p_pval = compute_pearson(predictions, labels)
    spearman, s_pval = compute_spearman(predictions, labels)
    rmse = compute_rmse(predictions, labels)
    #rint(acc," ", pearson, " ", p_pval, " ", spearman, " ", s_pval, " ", rmse)
    result = {
          #"acc": acc,
          "pearson": pearson,
          "pearson_p": p_pval,
          "spearman": spearman,
          "spearman_p": s_pval,
          "rmse":rmse
      }
    
    return result

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs) #loss, logits, layers
        logits = outputs[1]
        
        if labels is not None:
            loss_fct = MSELoss(device=model.device)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "model didnt return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs[0] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def load_nnif(nnif_path, indices, attacks, is_real=None):
    if_scores = np.ones((len(indices),max_indices))
    helpful_ranks= np.ones((len(indices),max_indices))
    helpful_dists = np.ones((len(indices),max_indices))
    harmful_ranks= np.ones((len(indices),max_indices))
    harmful_dists = np.ones((len(indices),max_indices))
    count=0
    for i, attack, real  in tqdm(zip(indices, attacks, is_real)):
        attack = "typo" if attack=="pruthi" or attack=="orig" else attack
        attack = "synonym" if attack=="alzantot" else attack
        
        all_neighbor_ranks = np.load(os.path.join(nnif_path, attack, 'all_neighbor_ranks.npy'))
        all_neighbor_dists = np.load(os.path.join(nnif_path, attack, 'all_neighbor_dists.npy'))
        all_neighbor_ranks_adv = np.load(os.path.join(nnif_path, attack, 'all_neighbor_ranks_adv.npy'))
        all_neighbor_dists_adv = np.load(os.path.join(nnif_path, attack, 'all_neighbor_dists_adv.npy'))

        if real==True:
            dir_ = os.path.join(nnif_path, "typo", '_index_{}'.format(i) ,"pred")
            case="pred"
        else:
            dir_ = os.path.join(nnif_path, attack, '_index_{}'.format(i) , "adv")
            case="adv"

        if os.path.exists(dir_):
            scores = np.load(os.path.join(dir_, 'scores.npy'))
            
            sorted_indices = np.argsort(scores)
            if case =="pred":
                ni   = all_neighbor_ranks
                nd   = all_neighbor_dists
            else:
                ni = all_neighbor_ranks_adv
                nd = all_neighbor_dists_adv

            #harmful = list(sorted_indices[:M])
            helpful = list(sorted_indices[-max_indices:][::-1])

            if_scores[count,:]=scores[helpful]
            
            #nnif[i,1,:]=scores[harmful]
            h_ranks, h_dists = find_ranks(i, sorted_indices[-max_indices:][::-1], ni, nd, mean=False)
            harm_ranks, harm_dists = find_ranks(i, sorted_indices[:max_indices], ni, nd, mean=False)
            helpful_ranks[count,:]=h_ranks
            helpful_dists[count,:]=h_dists
            harmful_ranks[count,:]=harm_ranks
            harmful_dists[count,:]=harm_dists
            count+=1

    if_scores = if_scores.reshape((if_scores.shape[0], -1))
    helpful_ranks = helpful_ranks.reshape((helpful_ranks.shape[0], -1))
    helpful_dists = helpful_dists.reshape((helpful_dists.shape[0], -1))
    harmful_ranks = harmful_ranks.reshape((harmful_ranks.shape[0], -1))
    harmful_dists = harmful_dists.reshape((harmful_dists.shape[0], -1))
    
    return if_scores, helpful_ranks, helpful_dists , harmful_ranks, harmful_dists
    
class CustomSMARTTrainer(Trainer):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
    
    def eval_func(self, embed, core_model):
        out = core_model(inputs_embeds=embed)
        res = out[0] if len(out)<2 else out[1]
        return res
    
    def compute_loss(self, model, inputs, return_outputs=False):        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs, output_hidden_states=True) #loss, logits, layers, hidden_states
        logits = outputs[1]

        if labels is not None:
            loss_fct = MSELoss(device=model.device)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "model didnt return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
                      
            #SMART
            #Compute initial (unperturbed) state
            if self.model_name == "smart_bert":
                core_model = self.model.bert
            if self.model_name == "smart_roberta":
                core_model = self.model.roberta
            embed = core_model.embeddings(inputs["input_ids"])
            state = self.eval_func(embed, core_model) #<- logits
            loss_fct =MSELoss()
            #loss = loss_fct(state.view(-1, 1), labels.view(-1, 1))#F.cross_entropy
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

            if self.is_in_train:
                # Define SMART loss (Regularizer)
                smart_loss_fn = SMARTLoss(eval_fn = self.eval_func, loss_fn = loss_fct, loss_last_fn = loss_fct, core_model = core_model) 
                
                #print(smart_loss_fn(embed, state))
                
                loss += 0.02 * smart_loss_fn(embed, state) #eqn 1 #self.weight #outputs[2][1]

        return (loss, outputs) if return_outputs else loss    

    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
      metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
      )
    config_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
      )
    tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
      default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field()
    column_info_path: str = field(default=None)
    column_info: dict = field(default=None,metadata={'help': 'a dict referencing the text, categorical, numerical, and label columns its keys are text_cols, num_cols, cat_cols, and label_col'})

    categorical_encode_type: str = field(default='ohe', metadata={'help': 'sklearn encoder to use for categorical data','choices': ['ohe', 'binary', 'label', 'none']})
    numerical_transformer_method: str = field(default='yeo_johnson', metadata={'help': 'sklearn numerical transformer to preprocess numerical data','choices': ['yeo_johnson', 'box_cox', 'quantile_normal', 'none']})
    task: str = field(default="classification", metadata={"help": "The downstream training task","choices": ["classification", "regression"]})

    mlp_division: int = field(default=4,metadata={'help': 'the ratio of the number of hidden dims in a current layer to the next MLP layer'})
    combine_feat_method: str = field(default='individual_mlps_on_cat_and_numerical_feats_then_concat', metadata={'help': 'method to combine categorical and numerical features, see README for all the method'})
    mlp_dropout: float = field(default=0.1,metadata={'help': 'dropout ratio used for MLP layers'})
    numerical_bn: bool = field(default=True,metadata={'help': 'whether to use batchnorm on numerical features'})
    use_simple_classifier: str = field(default=True,metadata={'help': 'whether to use single layer or MLP as final classifier'})
    mlp_act: str = field(default='relu',metadata={'help': 'the activation function to use for finetuning layers','choices': ['relu', 'prelu', 'sigmoid', 'tanh', 'linear']})
    gating_beta: float = field(default=0.2,metadata={'help': "the beta hyperparameters used for gating tabular data see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"})

    def __post_init__(self):
        assert self.column_info != self.column_info_path
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, 'r') as f:
                self.column_info = json.load(f)

def find_ranks(test_idx, sorted_influence_indices, all_neighbor_indices, all_neighbor_dists, mean=False):
    ni = all_neighbor_indices
    nd = all_neighbor_dists
    
    num_output = 1 #len(model.net)

    ranks = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.int32)
    dists = -1 * np.ones((num_output, len(sorted_influence_indices)), dtype=np.float32)
    
#     print(ni.shape)
#     print()
#     print(nd.shape)

    for target_idx in range(len(sorted_influence_indices)):
#         print(target_idx)
        idx = sorted_influence_indices[target_idx]
#         print(idx)
        
        #for layer_index in range(num_output):
        loc_in_knn = np.where(ni[test_idx, 0] == idx)[0][0]
#         print("loc_in_knn:", idx, loc_in_knn)
        knn_dist = nd[test_idx, 0, loc_in_knn]
#         print("knn_dist:", knn_dist)
        
        ranks[0, target_idx] = loc_in_knn
        dists[0, target_idx] = knn_dist *  10

    if mean:
        ranks_mean = np.mean(ranks, axis=1)
        dists_mean = np.mean(dists, axis=1)
        return ranks_mean, dists_mean
    
    ranks = ranks.flatten()
    dists = dists.flatten()
    
#     print(ranks)
#     print(dists)
#     sys.exit()
    
    
    return ranks, dists

def gen_columns(args, max_indices):
    col_names = ["text"]
    if args.nnif =='v':
        for j in range(0,max_indices):
            col_names.append("if_"+str(j))
        for j in range(0, max_indices):
            col_names.append("hprnk_"+str(j))
        for j in range(0, max_indices):
            col_names.append("hpdist_"+str(j))
        for j in range(0, max_indices):
            col_names.append("hrrnk_"+str(j))
        for j in range(0, max_indices):
            col_names.append("hrdist_"+str(j))
    if args.nnif =='m':
        col_names.append("inf")
        col_names.append("hprnk")
        col_names.append("hpdist")
        col_names.append("hrrnk")
        col_names.append("hrdist")
     
    #TODO
    #if args.lid:
    if args.bert_score == True:
        col_names.append("bert_score")
    if args.gpt:
        col_names.append("gpt")    
    if args.feats =='gm':
        col_names.append("gram")
    if args.ngrams is not None:
        col_names.append("ngram")
    if args.feats =='lang':
        col_names.append("lang")
        
    return col_names
                
if __name__ == '__main__':
    #<editor-fold desc="arguments">
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli', 'MR', 'TrustPilot'],
                        help='train') #True
    parser.add_argument('--dataset-path', type=str, required=True,
                        default="./data/data.csv", help='train dataset path.') #True
    parser.add_argument('--datafile', type=str, required=True,
                        default="data.csv", help='train dataset file.') #True
    parser.add_argument('--gramm-path', type=str, required=False,
                        default=None, help='The dataset file containing grammaticality scores.') #True
    # parser.add_argument('--nnif-path', type=str, required=False,
    #                     default=None, help='Folder containing IF scores.') #True
    parser.add_argument('--adv-path', type=str, required=False, default="./data/", help='The directory of the adversarial dataset.')
    parser.add_argument('--model-name', type=str, required=False, choices=['bert', 'roberta', 'xlnet', 'albert', 'distilbert', 'smart_bert', 'smart_roberta', 'sentence_bert'])
    #training features
    parser.add_argument('--nnif', default=None, type=str, required=False, choices=['v', 'm'])
    parser.add_argument('--lid',action="store_true")
    parser.add_argument('--gpt',action="store_true")
    parser.add_argument('--bert-score',action="store_true")
    parser.add_argument('--ngrams', default=None, type=int, required=False, choices=[1, 2, 3])
    parser.add_argument('--do-train',action="store_true")
    parser.add_argument('--do-eval',action="store_true")
    parser.add_argument('--eval-dataset-name', type=str, default=None, required=False, choices=['IMDB', 'Mnli', 'MR', 'TrustPilot'],
                        help='detecting which test set`s adversarial examples.') #True
    parser.add_argument('--eval-dataset-path', type=str, required=False,
                        default="./data/data.csv", help='The dataset file.') #True
    parser.add_argument('--feats', type=str, required=False, choices=['gm', 'lang'], default=None) 
    
    parser.add_argument('--model-dir', type=str, required=False,
                        default="./data/", help='The directory of the finetuned model.')
    parser.add_argument('--nnif-dir', type=str, required=False,
                        default="./results/", help='The directory of the nnif scores.')
    parser.add_argument('--attack', type=str, default=None, required=False, choices=['typo', 'synonym', 'textfooler', 'bae'],
                        help='Attack method to generate adversarial examples.')#True
    parser.add_argument('--max-length', type=int, default=None, required=False, choices=[768,512, 256, 128],
                        help='The maximum sequences length.')#True
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for transformer models.')#32
    parser.add_argument('--random-seed', type=int, default=38, help='random seed value.')
    
    #parser.add_argument('--start', type=int, default=0, help='starting test id')
    parser.add_argument('--end', type=int, default=0, help='ending test id')
    parser.add_argument('--damping', type=float, default=0.0, help="probably need damping for deep models")
    
    parser.add_argument('--checkpoint_dir', type=str, default='', required=False,
                        help='Checkpoint dir, the path to the saved model architecture and weights')
    parser.add_argument('--start', type=int, default=1, help='starting of the epochs.')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs.')
    parser.add_argument('--k', type=int, default=5, help='k-folds crossvalidation.')
    parser.add_argument('--model', type=str, default=None, required=False, choices=['bert', 'bert_nnif'])
    parser.add_argument('--max-indices', type=int, default=-1, help='maximum number of helpful indices to use in NNIF detection')
   


    args = parser.parse_args()

    batch_size = 5
    max_length = args.max_length
    #max_indices = args.max_indices

    # set a random seed value all over the place to make this reproducible.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # check if there's a GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        #print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
    
    #load model
    #['bert', 'roberta', 'xlnet', 'bart', 'smart_bert', 'smart_roberta']
    if args.model_name=='bert' or args.model_name=='smart_bert':
        print("BERT")
        model_params=bert_params
    if args.model_name=='roberta' or args.model_name=='smart_roberta':
        print("RoBERTa")
        model_params=roberta_params
    if args.model_name=='xlnet':
        print("XLNET")
        model_params=xlnet_params
    if args.model_name=='albert':
        print("AlBERT")
        model_params=albert_params
    if args.model_name=='distilbert':
        print("DistilBERT")
        model_params=distilbert_params        
    

    #print("Load datasets.")

    data = pd.read_csv(os.path.join(args.dataset_path, args.datafile)) 
    
#     if not os.path.exists(os.path.join(args.dataset_path, "train_indices.npy")):
#         all_indices = np.arange(len(data))
#         train_dev = random.sample(list(all_indices), int(0.9 * len(data)))
#         test_indices = np.setdiff1d(all_indices, train_dev)
#         train_indices = random.sample(list(train_dev), int(0.9 * len(train_dev)))
#         dev_indices = np.setdiff1d(train_dev, train_indices)
        
#         np.save(os.path.join(args.dataset_path, "train_indices.npy"), train_indices)
#         np.save(os.path.join(args.dataset_path, "dev_indices.npy"), dev_indices)
#         np.save(os.path.join(args.dataset_path, "test_indices.npy"), test_indices)
#     else:
#         train_indices = np.load(os.path.join(args.dataset_path, "train_indices.npy"))
#         dev_indices = np.load(os.path.join(args.dataset_path, "dev_indices.npy"))
#         test_indices = np.load(os.path.join(args.dataset_path, "test_indices.npy"))

    temp_indices = data[data["common"] == False].index
    train_indices = random.sample(list(temp_indices), int(0.9 * len(temp_indices)))
    dev_indices = np.setdiff1d(temp_indices, train_indices)
    test_indices = data[data["common"] == True].index
    
    texts = data['text'].to_numpy()
    text_pairs = data['reference'].to_numpy()
    sus_labels = data['suspicion.score'].astype(float).to_numpy()
    
    if args.feats=="gm":
        gram_data = pd.read_csv(args.gramm_path)
        gram = gram_data['g_count'].to_numpy()
        lang = None
    elif args.feats=="lang":
        gram_data = pd.read_csv(args.gramm_path)
        lang = gram_data['lang_count'].to_numpy()
        gram = None
    else:
        gram = None
        lang = None
    if args.bert_score == True:
        bert_scores = data["bertscore_f1"] 
    if args.gpt == True:
        gpt_scores = data["gpt_adj"] 
    
    if args.ngrams==1:
        n_grams = data["unigram"]
    elif args.ngrams==2:
        n_grams = data["bigram"]    
    elif args.ngrams==3:
        n_grams = data["trigram"]
    else:
        n_grams = None
        
    if args.attack is not None:
        output_dir = os.path.join('./results', args.dataset_name, args.model_name)
    else:
        output_dir = os.path.join('./results', args.dataset_name, args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    print("text")
    folder_name="text"
    if args.nnif is not None:
        print("NNIF")
        folder_name+="_nnif"+args.nnif
    if args.gpt:
        print("GPT4")
        folder_name+="_gpt"
    if args.bert_score == True:
        print("Bert scores")
        folder_name+="_bertscore"
    if args.lid:
        print("lid")
        folder_name+="_lid"
    if args.feats is not None:
        print("Grammaticality:", args.feats)
        folder_name+="_"+args.feats
    if args.ngrams is not None:
        print("Ngrams:", args.ngrams)
        folder_name+="_"+str(args.ngrams)+"_ngram"
    
    
    result_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(result_dir)    
    max_indices= 100 if args.max_indices == -1 else args.max_indices
    col_names = gen_columns(args, max_indices)
    
    num_labels = 1
    #print(col_names)
    
    print("Creatng dataset.")

    #combine text + nnif + lid + gram
    dataset = pd.DataFrame(columns = col_names)
    dataset["text"] = texts
    dataset["text_pair"] = text_pairs
    if args.feats =='gm':
        dataset["gram"] = gram
    if args.feats =='lang':
        dataset["lang"] = lang
    if args.gpt==True:
        dataset["gpt"] = gpt_scores
    if args.bert_score == True:
        dataset["bert_score"] = bert_scores
    if args.ngrams is not None:
        dataset["ngram"] = n_grams
    
    if args.nnif is not None:
        if not os.path.exists(os.path.join(args.dataset_path, 'if_scores.npy')):
            nnif_indices = data["nnif_ind"]
            attacks = data["group"]
            is_real = data["is_real"]

            nnif_data  = load_nnif(args.nnif_dir, nnif_indices, attacks, is_real)
            if_data, help_ranks, help_dists , harm_ranks, harm_dists = nnif_data
            #np.save(os.path.join(args.dataset_path, 'if_scores.npy'), if_data)
            np.save(os.path.join(args.dataset_path, 'helpful_ranks.npy'), help_ranks)
            np.save(os.path.join(args.dataset_path, 'helpful_dists.npy'), help_dists)
            np.save(os.path.join(args.dataset_path, 'harmful_ranks.npy'), harm_ranks)
            np.save(os.path.join(args.dataset_path, 'harmful_dists.npy'), harm_dists)
        else: 
            #if_data= np.load(os.path.join(args.dataset_path, 'if_scores.npy'))
            help_ranks=np.load(os.path.join(args.dataset_path, 'helpful_ranks.npy'))
            help_dists=np.load(os.path.join(args.dataset_path, 'helpful_dists.npy'))
            harm_ranks=np.load(os.path.join(args.dataset_path, 'harmful_ranks.npy'))
            harm_dists=np.load(os.path.join(args.dataset_path, 'harmful_dists.npy'))

        if args.nnif=='v':
            # for j in range(0, max_indices):
            #     dataset["if_"+str(j)] = if_data[:,j]
            for j in range(0, max_indices):
                dataset["hprnk_"+str(j)] = help_ranks[:,j]
            for j in range(0, max_indices):
                dataset["hpdist_"+str(j)] = help_dists[:,j]
            for j in range(0, max_indices):
                dataset["hrrnk_"+str(j)] = harm_ranks[:,j]
            for j in range(0, max_indices):
                dataset["hrdist_"+str(j)] = harm_dists[:,j]
        if args.nnif=='m':
            #dataset["inf"] = np.mean(if_data, axis=1).reshape(-1,1)
            dataset["hprnk"] = np.mean(help_ranks, axis=1).reshape(-1,1)
            dataset["hpdist"] = np.mean(help_dists, axis=1).reshape(-1,1)
            dataset["hrrnk"] = np.mean(harm_ranks, axis=1).reshape(-1,1)
            dataset["hrdist"] = np.mean(harm_dists, axis=1).reshape(-1,1)
         
    dataset['suspicion_score'] = sus_labels
    print("dataset preapared:", dataset.shape)
    
    
    # train_df, test_df = np.split(dataset.sample(frac=1), [int(.9*len(dataset))])
    # train_df, dev_df = np.split(train_df.sample(frac=1), [int(.9*len(train_df))])
    train_df = dataset.iloc[list(train_indices), :]
    test_df = dataset.iloc[list(test_indices), :]
    dev_df = dataset.iloc[list(dev_indices), :]
    
    #train_df=train_df.iloc[0:100, :]
    #dev_df = dev_df.iloc[0:10, :]
    #test_df = test_df.iloc[0:10,:]
        
    train_df.to_csv(result_dir+'/train.csv', index=True)
    dev_df.to_csv(result_dir+'/dev.csv', index=True)
    test_df.to_csv(result_dir+'/test.csv', index=True)
    

    
    print("train:", train_df.shape, " dev:",dev_df.shape, " test:",test_df.shape)
    
    text_cols = ['text', 'text_pair']
    numerical_cols = col_names[1:] if len(col_names)>1 else None
    column_info_dict = {
        'text_cols': text_cols,
        'num_cols': numerical_cols,
        'cat_cols': None,
        'label_col': 'suspicion_score' 
        #'label_list': ['Not Recommended', 'Recommended']
    }

    model_dir = model_params['pretrained_file_path']
    model_args = ModelArguments( model_name_or_path=model_dir)#
    data_args = MultimodalDataTrainingArguments(data_path=result_dir, combine_feat_method='concat', column_info=column_info_dict, task='regression')
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=model_args.cache_dir, max_length=max_length)
    
    # if len(col_names)==1:
    #     transformer_method=None
    # else:
    #     transformer_method="quantile_normal"
    transformer_method="quantile_normal"
    train_dataset, dev_dataset, test_dataset = load_data_from_folder(
        data_args.data_path,
        data_args.column_info['text_cols'],
        tokenizer,
        label_col=data_args.column_info['label_col'],
        label_list=None,
        numerical_cols=data_args.column_info['num_cols'],
        sep_text_token_str=tokenizer.sep_token,
        max_token_length = max_length,
        numerical_transformer_method = transformer_method
    )
    
    config = AutoConfig.from_pretrained( model_args.config_name if model_args.config_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir,)
    
    if len(col_names)==1:
        tabular_config = TabularConfig(num_labels=num_labels, cat_feat_dim=None, numerical_feat_dim=None, **vars(data_args))
    else:
        tabular_config = TabularConfig(num_labels=num_labels, cat_feat_dim=None, numerical_feat_dim=train_dataset.numerical_feats.shape[1], **vars(data_args))
    config.tabular_config = tabular_config
    
    model = AutoModelWithTabular.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, config=config, cache_dir="./cache")
    model.to(device)

    
    epochs = [1,2,3,4,5]#,6,7,8,9,10,15,20, 25]
    if args.do_train:        
        for epoch in epochs:
            print(f'\nEPOCH: {epoch}')
            training_args = TrainingArguments(
                    output_dir=output_dir,
                    logging_dir=None,
                    overwrite_output_dir=True,
                    do_train=True,
                    do_eval=True,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=epoch,
                    #evaluate_during_training=False,
                    logging_steps=25,
                    # eval_steps=250,
                    learning_rate = model_params['learning_rate'],
                    #weight_decay=0.01,
                    # evaluation_strategy="epoch",
                    save_strategy="no",
                    #load_best_model_at_end=True,
            )
            set_seed(training_args.seed)
            print('Training and validation model ... ')
            start= time.time()

            #default optimizer is AdamW with a get_linear_schedule_with_warmup() schedule
            #print("model:", model)
            total_steps = int(len(train_dataset)/4) // epoch
            optimizer = optim.AdamW(model.parameters(), lr=model_params['learning_rate'], eps=1e-08)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_steps * 0.06,
                                                num_training_steps=total_steps)
            
            print("total steps:", total_steps)
            
            if args.model_name=='smart_bert' or args.model_name=='smart_roberta': 
                trainer = CustomSMARTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    compute_metrics=calc_classification_metrics,
                    # callbacks=[EarlyStoppingCallback(total_steps)],
                    tokenizer = tokenizer,
                    #optimizers = [optimizer, scheduler],
                    model_name = args.model_name,
                )
            else:
                trainer = CustomTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=dev_dataset,
                    compute_metrics=calc_classification_metrics,
                    # callbacks=[EarlyStoppingCallback(total_steps)],
                    tokenizer = tokenizer,
                    #optimizers = [optimizer, scheduler],
                )
            
            #print(trainer)
            trainer.train()
            metrics = trainer.evaluate()
            #print("eval:", metrics)
            #print("save model:", result_dir)
            #torch.save(trainer.model.state_dict(), os.path.join(result_dir, 'model.pt'))
            #tokenizer.save_pretrained(result_dir)
            #trainer.save_model(result_dir)
            predictions, labels, metrics = trainer.predict(train_dataset)
            print("train:", metrics)
            for s,p in zip(train_df['suspicion_score'], predictions[0]):
                print(s, ",", p)
            predictions, labels, metrics = trainer.predict(dev_dataset)
            print("dev:", metrics)
            for s,p in zip(dev_df['suspicion_score'], predictions[0]):
                print(s, ",", p)
            predictions, labels, metrics = trainer.predict(test_dataset)
            print("test:", metrics)
            for s,p in zip(test_df['suspicion_score'], predictions[0]):
                print(s, ",", p)
        end_test = time.time()
        hours, rem = divmod(end_test - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('Training runtime: {:0>2}:{:0>2}:{:0>2f}'.format(int(hours),int(minutes),seconds))