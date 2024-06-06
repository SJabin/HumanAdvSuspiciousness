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
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge,LinearRegression,HuberRegressor, LogisticRegression, ElasticNet

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW, RMSprop, SGD
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, KFold
from sklearn.linear_model import LogisticRegressionCV
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, XLNetTokenizer, XLNetModel, BartTokenizer, BartModel
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr, spearmanr

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

def gen_columns(args, folder_name, max_indices):
    col_names = []
    if args.nnif is not None:
        #print("NNIF", args.nnif)
        folder_name+="_nnif"+args.nnif
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
            col_names.append("if")
            col_names.append("hprnk")
            col_names.append("hpdist")
            col_names.append("hrrnk")
            col_names.append("hrdist")
    if args.gpt:
        #print("GPT4")
        folder_name+="_gpt"
        col_names.append("gpt")
    if args.lid:
        #print("lid")
        folder_name+="_lid"
        col_names.append("lid1")
        col_names.append("lid2")
        col_names.append("lid3")
        col_names.append("lid4")
        col_names.append("lid5")
        col_names.append("lid6")
        col_names.append("lid7")
        col_names.append("lid8")
        col_names.append("lid9")
        col_names.append("lid10")
        col_names.append("lid11")
        col_names.append("lid12")
        col_names.append("lid")
    if args.gram:
        #print("Gramfomer")
        folder_name+="_gram"
        col_names.append("gram")
        
    if args.lang:
        #print("LangTool")
        folder_name+="_lang"
        col_names.append("lang")
    
    if args.pert:
        #print("PertRate")
        folder_name+="_pertrate"
        col_names.append("pert_rate")

    if args.ngrams:
        #print("NGrams")
        folder_name+="_ngrams"
        col_names.append("unigr_overlap")
        col_names.append("bigr_overlap")
        col_names.append("trigr_overlap")
        col_names.append("unigr_diff")
        col_names.append("bigr_diff")
        col_names.append("trigr_diff")
    if args.paired:
        #print("PairedScores")
        folder_name+="_paired"
        col_names.append("bertscore_p")
        col_names.append("bertscore_r")
        col_names.append("bertscore_f1")
        # col_names.append("unigr_diff")
        # col_names.append("bigr_diff")
        # col_names.append("trigr_diff")
        #col_names.append("sbert_sim")
        col_names.append("use_sim")
        col_names.append("meaning_bert")
        col_names.append("lavenshtein")
        col_names.append("hamming")
        #col_names.append("laven_token")
        #col_names.append("hamming_token")
        col_names.append("bow_norm")
        #col_names.append("bow_token")
        #col_names.append("glove_diff")
        col_names.append("bleu_score")
        col_names.append("meteor")
        col_names.append("rouge1")
        col_names.append("rouge2")
        col_names.append("rouge3")
        col_names.append("rougeL")
        col_names.append("bleurt")
    if args.readability==True:
        col_names.append("Flesch_kinncaid")
        col_names.append("flesch")
        col_names.append("gunning_fog")
        col_names.append("coleman_liau")
        col_names.append("dale_chall")
        col_names.append("ari")
        col_names.append("linsear_write")
        col_names.append("spache")
        col_names.append("orig_flesch_kinncaid")
        col_names.append("orig_flesch")
        col_names.append("orig_gunning_fog")
        col_names.append("orig_coleman_liau")
        col_names.append("orig_dale_chall")
        col_names.append("ori_ari")
        col_names.append("ori_linsear_write")
        col_names.append("ori_spache")
        
    return col_names, folder_name
                
if __name__ == '__main__':
    #<editor-fold desc="arguments">
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default=None, required=True, choices=['IMDB', 'Mnli', 'MR', 'TrustPilot'],
                        help='train') #True
    parser.add_argument('--dataset-path', type=str, required=True,
                        default="./data/data.csv", help='train dataset path.') #True
    parser.add_argument('--transform', type=str, required=False,
                        default="quantile_normal", choices =['yeo_johnson', 'box_cox', 'quantile_normal', 'minmaxscalar'])
    parser.add_argument('--label-trans', action="store_true")

    parser.add_argument('--datafile', type=str, required=True,
                        default="data.csv", help='train dataset file.') #True
    parser.add_argument('--embed_file', type=str, required=True,
                        default="glove.npy", help='word embeddings for the texts.') #True
    parser.add_argument('--orig_embed_file', type=str, required=True,
                        default="glove.npy", help='word embeddings for the originall texts.') #True
    parser.add_argument('--gramm-path', type=str, required=False,
                        default=None, help='The dataset file containing grammaticality scores.') #True
    # parser.add_argument('--nnif-path', type=str, required=False,
    #                     default=None, help='Folder containing IF scores.') #True
    parser.add_argument('--adv-path', type=str, required=False, default="./data/", help='The directory of the adversarial dataset.')
    parser.add_argument('--model-name', type=str, required=False, choices=['bert', 'roberta', 'xlnet', 'albert', 'distilbert', 'smart_bert', 'smart_roberta', 'sentence_bert', 'ml'])
    #training features
    parser.add_argument('--nnif', default=None, type=str, required=False, choices=['v', 'm'])
    parser.add_argument('--embed', default=None, type=str, required=False, choices=['glove', 'word2vec'])
    parser.add_argument('--lid',action="store_true")
    parser.add_argument('--gpt',action="store_true")
    parser.add_argument('--gram',action="store_true")
    parser.add_argument('--lang',action="store_true")
    parser.add_argument('--pert',action="store_true")
    parser.add_argument('--ngrams',action="store_true")
    parser.add_argument('--readability',action='store_true')
    parser.add_argument('--paired',action="store_true")
    parser.add_argument('--do-train',action="store_true")
    parser.add_argument('--do-eval',action="store_true")
    parser.add_argument('--eval-dataset-name', type=str, default=None, required=False, choices=['IMDB', 'Mnli', 'MR', 'TrustPilot'],
                        help='detecting which test set`s adversarial examples.') #True
    parser.add_argument('--eval-dataset-path', type=str, required=False,
                        default="./data/data.csv", help='The dataset file.') #True
    #parser.add_argument('--feats', type=str, required=False, choices=['gm', 'lang'], default=None) 
        
    parser.add_argument('--model-dir', type=str, required=False,
                        default="./data/", help='The directory of the finetuned model.')
    parser.add_argument('--nnif-dir', type=str, required=False,
                        default="./results/", help='The directory of the nnif scores.')
    parser.add_argument('--attack', type=str, default=None, required=False, choices=['typo', 'synonym', 'textfooler', 'bae'],
                        help='Attack method to generate adversarial examples.')#True
    parser.add_argument('--max-length', type=int, default=None, required=False, choices=[768,512, 256, 128,64],
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
        
    data = pd.read_csv(os.path.join(args.dataset_path, args.datafile)) 
    
    temp_indices = data[data["common"] == False].index
    train_indices = random.sample(list(temp_indices), int(0.9 * len(temp_indices)))
    dev_indices = np.setdiff1d(temp_indices, train_indices)
    test_indices = data[data["common"] == True].index
    
    # if not os.path.exists(os.path.join(args.dataset_path,args.dataset_name,"train_idx_regressor.npy")):
    #     temp_indices = data.index
    #     train_dev_indices = random.sample(list(temp_indices), int(0.9 * len(temp_indices)))
    #     test_indices = np.setdiff1d(temp_indices, train_dev_indices)
    #     train_indices = random.sample(list(train_dev_indices), int(0.9 * len(train_dev_indices)))
    #     dev_indices = np.setdiff1d(train_dev_indices, train_indices)
    #     np.save(os.path.join(args.dataset_path, 'train_idx_regressor.npy'), train_indices)
    #     np.save(os.path.join(args.dataset_path, 'dev_idx_regressor.npy'), dev_indices)
    #     np.save(os.path.join(args.dataset_path, 'test_idx_regressor.npy'), test_indices)
    # else:
    #     train_indices = np.load(os.path.join(args.dataset_path,args.dataset_name, 'train_idx_regressor.npy'))
    #     dev_indices = np.load(os.path.join(args.dataset_path,args.dataset_name, 'dev_idx_regressor.npy'))
    #     test_indices = np.load(os.path.join(args.dataset_path,args.dataset_name, 'test_idx_regressor.npy'))
    
    if args.attack is not None:
        output_dir = os.path.join('./results', args.dataset_name, args.model_name)
    else:
        output_dir = os.path.join('./results', args.dataset_name, args.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
       
    max_indices= 100 if args.max_indices == -1 else args.max_indices
    col_names, folder_name = gen_columns(args, "ML", max_indices)
    result_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print(result_dir)
    
     #texts = data['text'].to_numpy() 
    sus_labels = data['suspicion.score'].astype(float).to_numpy() if not args.label_trans else data['log.score'].astype(float).to_numpy()    
        
    
    num_labels = 1
    print(col_names)
    
    print("Creatng dataset.")
    
    
    dataset = np.load(args.embed_file)
    
    if args.paired == True:
        orig_embed = np.load(args.orig_embed_file)
        dataset = np.c_[dataset, orig_embed]
    # dataset = np.load("C:/Users/45054541/Projects/Human_suspicion/Regressor/glove_single.npy")
    # word2vec = np.load("C:/Users/45054541/Projects/Human_suspicion/Regressor/word2vec_single.npy")
    # dataset = np.c_[dataset, word2vec]

    #combine text + nnif + lid + gram
    #dataset = pd.DataFrame(columns = col_names)
    if args.gram:
        dataset = np.c_[dataset, data['g_count'].to_numpy()]
    if args.lang:
        dataset= np.c_[dataset, data['lang_count'].to_numpy()]
    if args.gpt==True:
        dataset= np.c_[dataset, data["gpt_adj"].to_numpy()]
    if args.lid==True:
        dataset = np.c_[dataset, data["lid1"].to_numpy()]
        dataset = np.c_[dataset, data["lid2"].to_numpy()]
        dataset= np.c_[dataset, data["lid3"].to_numpy()] 
        dataset = np.c_[dataset, data["lid4"].to_numpy()]
        dataset = np.c_[dataset, data["lid5"].to_numpy()] 
        dataset= np.c_[dataset, data["lid6"].to_numpy()] 
        dataset = np.c_[dataset, data["lid7"].to_numpy()]
        dataset = np.c_[dataset, data["lid8"].to_numpy()] 
        dataset= np.c_[dataset, data["lid9"].to_numpy()]
        dataset= np.c_[dataset, data["lid10"].to_numpy()] 
        dataset = np.c_[dataset, data["lid11"].to_numpy()] 
        dataset= np.c_[dataset, data["lid12"].to_numpy()] 
        dataset= np.c_[dataset, data["lid"].to_numpy()] 
    
    
    if args.nnif is not None:
        if not os.path.exists(os.path.join(args.dataset_path, 'if_scores.npy')):
            nnif_indices = data["nnif_ind"].to_numpy()
            attacks = data["group"].to_numpy()
            is_real = data["is_real"].to_numpy()

            nnif_data  = load_nnif(args.nnif_dir, nnif_indices, attacks, is_real)
            if_data, help_ranks, help_dists , harm_ranks, harm_dists = nnif_data
            np.save(os.path.join(args.dataset_path, 'trustpilot_if_scores.npy'), if_data)
            np.save(os.path.join(args.dataset_path, 'trustpilot_helpful_ranks.npy'), help_ranks)
            np.save(os.path.join(args.dataset_path, 'trustpilot_helpful_dists.npy'), help_dists)
            np.save(os.path.join(args.dataset_path, 'trustpilot_harmful_ranks.npy'), harm_ranks)
            np.save(os.path.join(args.dataset_path, 'trustpilot_harmful_dists.npy'), harm_dists)
        else: 
            if_data= np.load(os.path.join(args.dataset_path, 'trustpilot_if_scores.npy'))
            help_ranks=np.load(os.path.join(args.dataset_path, 'trustpilot_helpful_ranks.npy'))
            help_dists=np.load(os.path.join(args.dataset_path, 'trustpilot_helpful_dists.npy'))
            harm_ranks=np.load(os.path.join(args.dataset_path, 'trustpilot_harmful_ranks.npy'))
            harm_dists=np.load(os.path.join(args.dataset_path, 'trustpilot_harmful_dists.npy'))

        if args.nnif=='v':
            for j in range(0, max_indices):
                dataset= np.c_[dataset, if_data[:,j]]
            for j in range(0, max_indices):
                dataset = np.c_[dataset, help_ranks[:,j]]
            for j in range(0, max_indices):
                dataset = np.c_[dataset, help_dists[:,j]]
            for j in range(0, max_indices):
                dataset= np.c_[dataset, harm_ranks[:,j]]
            for j in range(0, max_indices):
                dataset= np.c_[dataset, harm_dists[:,j]]
        if args.nnif=='m':
            dataset = np.c_[dataset, np.mean(if_data, axis=1).reshape(-1,1)]
            dataset= np.c_[dataset, np.mean(help_ranks, axis=1).reshape(-1,1)]
            dataset = np.c_[dataset, np.mean(help_dists, axis=1).reshape(-1,1)]
            dataset= np.c_[dataset, np.mean(harm_ranks, axis=1).reshape(-1,1)]
            dataset = np.c_[dataset, np.mean(harm_dists, axis=1).reshape(-1,1)]
    if args.pert:
        dataset= np.c_[dataset, data["pert_rate"]]

    if args.ngrams:
        dataset = np.c_[dataset, data["unigr_overlap"]]
        dataset = np.c_[dataset, data["bigr_overlap"]]
        dataset= np.c_[dataset, data["trigr_overlap"]]
        dataset= np.c_[dataset, data["unigr_diff"]]
        dataset= np.c_[dataset, data["bigr_diff"]]
        dataset= np.c_[dataset, data["trigr_diff"]]
    if args.paired:
        dataset= np.c_[dataset, data["bertscore_p"]]
        dataset= np.c_[dataset, data["bertscore_r"]]
        dataset= np.c_[dataset, data["bertscore_f1"]]
        #dataset= np.c_[dataset, data["sbert_sim"]]
        dataset= np.c_[dataset, data["use_sim"]]
        dataset= np.c_[dataset, data["meaning_bert"]]
        dataset= np.c_[dataset, data["lavenshtein"]]
        dataset= np.c_[dataset, data["hamming"]]
        # dataset= np.c_[dataset, data["laven_token"]]
        # dataset= np.c_[dataset, data["hamming_token"]]
        dataset= np.c_[dataset, data["bow_norm"]]
        #dataset= np.c_[dataset, data["bow_token"]]
        # dataset= np.c_[dataset, data["word2vec_diff"]]
        # dataset= np.c_[dataset, data["glove_diff"]]
        dataset= np.c_[dataset, data["bleu_score"]]
        dataset= np.c_[dataset, data["meteor"]]
        dataset= np.c_[dataset, data["rouge1"]]
        dataset= np.c_[dataset, data["rouge2"]]
        dataset= np.c_[dataset, data["rouge3"]]
        dataset= np.c_[dataset, data["rougeL"]]
        dataset= np.c_[dataset, data["bleurt"]]
    if args.readability==True:
        dataset= np.c_[dataset, data["flesch_kinncaid"]]
        dataset= np.c_[dataset, data["flesch"]]
        dataset= np.c_[dataset, data["gunning_fog"]]
        dataset= np.c_[dataset, data["coleman_liau"]]
        dataset= np.c_[dataset, data["dale_chall"]]
        dataset= np.c_[dataset, data["ari"]]
        dataset= np.c_[dataset, data["linsear_write"]]
        dataset= np.c_[dataset, data["spache"]]
        dataset= np.c_[dataset, data["orig_flesch_kinncaid"]]
        dataset= np.c_[dataset, data["orig_flesch"]]
        dataset= np.c_[dataset, data["orig_gunning_fog"]]
        dataset= np.c_[dataset, data["orig_coleman_liau"]]
        dataset= np.c_[dataset, data["orig_dale_chall"]]
        dataset= np.c_[dataset, data["orig_ari"]]
        dataset= np.c_[dataset, data["orig_linsear_write"]]
        dataset= np.c_[dataset, data["orig_spache"]]
    print("dataset preapared:", dataset.shape)
    

    #Transform data
    # qt = QuantileTransformer(n_quantiles=10, random_state=0)
    # dataset = qt.fit_transform(dataset)
    # pt = PowerTransformer(method="yeo-johnson")
    # pt.fit(dataset)
    # dataset = pt.transform(dataset)
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)    
    dataset = np.c_[dataset, sus_labels]
    
    
    # train_df, test_df = np.split(dataset.sample(frac=1), [int(.9*len(dataset))])
    # train_df, dev_df = np.split(train_df.sample(frac=1), [int(.9*len(train_df))])
    train_df = dataset[list(train_indices), :]
    test_df = dataset[list(test_indices), :]
    dev_df = dataset[list(dev_indices), :]
    #dataset.to_csv(result_dir+'/dataset.csv', index=True)
    
    # train_df=train_df[0:100, :]
    # dev_df = dev_df[0:10, :]
    # test_df = test_df[0:10,:]
    
    # train_df.to_csv(result_dir+'/train.csv', index=True)
    # dev_df.to_csv(result_dir+'/dev.csv', index=True)
    # test_df.to_csv(result_dir+'/test.csv', index=True)
    
    print("train:", train_df.shape, " dev:", dev_df.shape," test:",test_df.shape)
    
    
    params = {
    "n_estimators": 400,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.1,
    "loss": "squared_error",
    }

    reg1 = GradientBoostingRegressor(random_state=42,**params)
    reg2 = RandomForestRegressor(random_state=42, max_depth=6, n_estimators=1500, max_features=1.0)
    reg3 = xgb.XGBRegressor(random_state=42)
    reg4 = LinearRegression()
    reg5 = HuberRegressor(max_iter = 10000)
    
    print("\nVotingRegressor")
    # train pearson's: 0.962 p_val: 0.000 spearman's: 0.950 p_val: 0.000 rmse:0.128 
    # dev pearson's: 0.130 p_val: 0.032 spearman's: 0.127 p_val: 0.038 rmse:0.241
    # test pearson's: 0.210 p_val: 0.000 spearman's: 0.191 p_val: 0.001 rmse:0.228
    ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('xg', reg3), ('lr', reg4), ('hr', reg5)])
    ereg = ereg.fit(train_df[:, :-1], train_df[:, -1])
    
    y_train_pred = ereg.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

    
    y_val_pred = ereg.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = ereg.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    
    print("\nStackingRegressor")
    estimators = [('gb', reg1), ('rf', reg2), ('xg', reg3), ('lr', reg4), ('hr', reg5)]
    ereg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
    ereg = ereg.fit(train_df[:, :-1], train_df[:, -1])
    
    y_train_pred = ereg.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

    
    y_val_pred = ereg.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = ereg.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    

    print("\nTrain GradientBoostingRegressor")
    gb_reg = GradientBoostingRegressor(**params)
    gb_reg.fit(train_df[:, :-1], train_df[:, -1])
    
    # Validate GradientBoostingRegressor
    y_train_pred = gb_reg.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_val_pred = gb_reg.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = gb_reg.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    
    print("\nRandomForestRegressor")
    rf_regressor = RandomForestRegressor(random_state=42, max_depth=6, n_estimators=1500, max_features=1.0)
    rf_regressor.fit(train_df[:, :-1], train_df[:, -1])
    
    # y_train_pred = rf_regressor.predict(dataset[:, :-1])
    # for p, l in zip(y_train_pred, dataset[:, -1]):
    #     print(l,",",p)
    
    
    # Validate RandomForestRegressor
    y_train_pred = rf_regressor.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_val_pred = rf_regressor.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = rf_regressor.predict(test_df[:, :-1])
    for p, l in zip(y_test_pred, test_df[:, -1]):
        print(l,",",p)
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

    
    print("\nTrain XGBoostRegressor")
    xgb_regressor =  xgb.XGBRegressor(random_state=42)
    xgb_regressor.fit(train_df[:, :-1], train_df[:, -1])

    # Validate RandomForestRegressor
    y_train_pred = xgb_regressor.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_val_pred = xgb_regressor.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = xgb_regressor.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

    
    print("\nTrain LinearRegression")
    linear_regressor = LinearRegression()
    linear_regressor.fit(train_df[:, :-1], train_df[:, -1])
    
    y_train_pred = linear_regressor.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_val_pred = linear_regressor.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = linear_regressor.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

    print("\nTrain HuberRegressor")
    huber_regressor = HuberRegressor(max_iter = 10000)
    huber_regressor.fit(train_df[:, :-1], train_df[:, -1])

    y_train_pred = huber_regressor.predict(train_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_train_pred, train_df[:, -1])
    spearman, spearman_p = compute_spearman(y_train_pred, train_df[:, -1] )
    rmse =  compute_rmse(y_train_pred, train_df[:, -1])  
    print(f"train pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_val_pred = huber_regressor.predict(dev_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_val_pred, dev_df[:, -1])
    spearman, spearman_p = compute_spearman(y_val_pred, dev_df[:, -1] )
    rmse =  compute_rmse(y_val_pred, dev_df[:, -1])  
    print(f"dev pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")
    
    y_test_pred = huber_regressor.predict(test_df[:, :-1])
    pearson, pearson_p = compute_pearson(y_test_pred, test_df[:, -1])
    spearman, spearman_p = compute_spearman(y_test_pred, test_df[:, -1] )
    rmse =  compute_rmse(y_test_pred, test_df[:, -1])  
    print(f"test pearson's: {pearson:.3f} p_val: {pearson_p} spearman's: {spearman:.3f} p_val: {spearman_p} rmse:{rmse:.3f}")

