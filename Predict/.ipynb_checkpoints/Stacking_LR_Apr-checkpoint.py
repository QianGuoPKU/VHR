import os 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import pickle
import json
import transformers
from pathlib import Path

import configparser
from transformers import bioinfo_compute_metrics as compute_metrics


import warnings
warnings.filterwarnings('ignore')


upper, lower = 1, 0
THRESHOLD = 0.5

def np_relu(x):
    return (x > 0) * x

def violation_process(hierarchy_structure, probs):
    
    taxa_list = list(hierarchy_structure.keys())
    taxa_roots = range(min(taxa_list))
    
    output = np.zeros(probs.shape)
    
    output[:,taxa_roots] = probs[:,taxa_roots]
    
    for child in taxa_list:
        parent = hierarchy_structure[child]
        
        p_child = probs[:,child]
        p_parent = output[:,parent]
        output[:,child] = p_child - np_relu(p_child - p_parent) 
        
    return output

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    ###### 1. training hyperparameters
    parser.add_argument(
        "--trained_model",
        type=str,
        help="Load trained model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="number of trained base model",
    )
    ###### 2. data
    parser.add_argument(
        "--base_path",
        type=str,
        help="Stacking model need input Base model results",
    )
    parser.add_argument(
        "--base_model_list",
        type=str,
        help="Base model list",
    )
    parser.add_argument(
        "--train_prob",
        type=str,
        help="output probs of training data in base model",
    )
    parser.add_argument(
        "--train_label",
        type=str,
        help="train data label",
    )
    parser.add_argument(
        "--valid_prob",
        type=str,
        help="output probs of test data in base model",
    )
    parser.add_argument(
        "--valid_label",
        type=str,
        help="test data label",
    )
    parser.add_argument(
        "--LABELS",
        type=str,
        help="test data label",
    )
    parser.add_argument(
        "--hierarchy_structure", 
        type=str, 
        help="Give hierarchy structure path"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        help="output file save path"
    )
    
    args = parser.parse_args()
    return args
    
    
def main():
    
    args = parse_args()  

    os.chdir(args.base_path)
    with open(args.valid_prob, 'rb') as handle:
        test_prob_agg = pickle.load(handle)

    with open(args.base_model_list, 'rb') as handle:    
        MODEL_list = pickle.load(handle)

    with open(args.LABELS, 'rb') as handle:    
        LABELS = pickle.load(handle)

    def jsonKeys2int(x):
        if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
        return x
    def separate_func(X, LABELS):
        X_full, X_miss = {}, {}
        full_index, miss_index = {}, {}
        for HOST in LABELS:
            X_full[HOST] = X[HOST][X[HOST]['Phylo'] != 0 ].values
            full_index[HOST] = X[HOST][X[HOST]['Phylo'] != 0 ].index.tolist()

            X_miss[HOST] = X[HOST][X[HOST]['Phylo'] == 0 ].iloc[:, 0:4].values
            miss_index[HOST] = X[HOST][X[HOST]['Phylo'] == 0 ].index.tolist()


        return X_full, full_index, X_miss, miss_index


    fj = open(args.hierarchy_structure)
    hierarchy_structure = json.load(fj)
    hierarchy_structure = jsonKeys2int(hierarchy_structure)
    
    if args.base_model != 'five_base':
        
        with open(args.trained_model + '/LR_model.pkl', 'rb') as handle:   
            Learners = pickle.load(handle)
    else:
        with open(args.trained_model + '/full_base/LR_model.pkl', 'rb') as handle:   
            full_Learners = pickle.load(handle)
        
        with open(args.trained_model + '/miss_base/LR_model.pkl', 'rb') as handle:   
            miss_Learners = pickle.load(handle)
    
    ########################
    
    X_Holdout = {}   
    eval_probs = {}
    '''测试集'''
    for HOST in LABELS:
        X_Holdout[HOST] = pd.DataFrame()
        for MODEL in MODEL_list:
            X_Holdout[HOST] = pd.concat([X_Holdout[HOST], test_prob_agg[MODEL][HOST]], axis =1)
 
        X_Holdout[HOST].columns = MODEL_list
        

     
        
        
    X_full, full_index, X_miss, miss_index = separate_func(X_Holdout, LABELS)
    
    for HOST in LABELS:
        if X_full[HOST].shape[0] != 0 :
            full_probs = full_Learners[HOST].predict(X_full[HOST])
        else:
            full_probs = X_full[HOST]
            
        if X_miss[HOST].shape[0] != 0 :
            miss_probs = miss_Learners[HOST].predict(X_miss[HOST])
        else:
            miss_probs = X_miss[HOST]  
        
        eval_probs[HOST] = np.zeros(len(X_Holdout[HOST]))
        if len(full_index[HOST])!=0:
            eval_probs[HOST][full_index[HOST]] = full_probs
        if len(miss_index[HOST])!=0:
            eval_probs[HOST][miss_index[HOST]] = miss_probs
    
    eval_probs = pd.DataFrame(eval_probs)
        
    updated_prob_df =  violation_process(hierarchy_structure, eval_probs.values)
    updated_prob_df[updated_prob_df<0] = 0
    updated_prob_df[updated_prob_df>1] = 1    
    
    eval_probs = updated_prob_df
    
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(eval_probs, columns = LABELS).to_csv(args.save_path + 'pred_probs.csv', index=False)
        
        
if __name__ == "__main__":
    main()