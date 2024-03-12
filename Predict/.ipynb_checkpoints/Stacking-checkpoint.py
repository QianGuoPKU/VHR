import os 
import argparse
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import math
import pickle
import json
import transformers

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

import configparser
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import AdamW, set_seed
from transformers import bioinfo_compute_metrics as compute_metrics


import warnings
warnings.filterwarnings('ignore')
### import Network
from mhc_model import MHC_Net, Flat_Net
CUDA_VISIBLE_DEVICES=0,1,2,3

upper, lower = 1, 0
THRESHOLD = 0.5


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
        
    # sum for all the labels 
    return output


def evaluate(model, eval_dataloader, args):

    model.eval()
    epoch_loss = 0
    logits = None
    
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            feature=batch[0]
            labels=batch[1]
            weights=batch[2]
            outputs = model(feature, labels, weights)
        
        if logits is None:
            logits = outputs['logits'].detach().cpu().numpy()
            reference = labels.detach().cpu().numpy()
        else:
            logits=np.append(logits,outputs['logits'].detach().cpu().numpy(), axis=0)
            reference=np.append(reference,labels.detach().cpu().numpy(), axis=0)
        
        loss = outputs['loss']
        epoch_loss += loss.item()
    
    eval_loss = epoch_loss/len(eval_dataloader)
    
    probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    
    ### post_processing of hierarchical violation
    f = open(args.hierarchy_structure)
    taxa = json.load(f)
    taxa = {int(k):int(v) for k,v in taxa.items()}
    probs = violation_process(taxa, probs)
    
    preds = np.where(probs > THRESHOLD, upper, lower)
    result = compute_metrics("mhc", preds, reference, probs)
    
    return eval_loss, result, probs

def predict(model, eval_dataloader, args):

    model.eval()
    epoch_loss = 0
    logits = None
    
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            feature=batch[0]
            labels=batch[1]
            weights=batch[2]
            outputs = model(feature, labels, weights)
        
        if logits is None:
            logits = outputs['logits'].detach().cpu().numpy()
            reference = labels.detach().cpu().numpy()
        else:
            logits=np.append(logits,outputs['logits'].detach().cpu().numpy(), axis=0)
            reference=np.append(reference,labels.detach().cpu().numpy(), axis=0)
        
        loss = outputs['loss']
        epoch_loss += loss.item()
    
    eval_loss = epoch_loss/len(eval_dataloader)
    
    probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    
    ### post_processing of hierarchical violation
    f = open(args.hierarchy_structure)
    taxa = json.load(f)
    taxa = {int(k):int(v) for k,v in taxa.items()}
    probs = violation_process(taxa, probs)

    preds = np.where(probs > THRESHOLD, upper, lower)
    result = compute_metrics("mhc", preds, reference, probs)
    
    return eval_loss, result, probs

def predict_no_label(model, eval_dataloader, args):
    model.eval()
    logits = None
    
    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            feature=batch[0]
            outputs = model(feature)
        
        if logits is None:
            logits = outputs['logits'].detach().cpu().numpy()
        else:
            logits=np.append(logits,outputs['logits'].detach().cpu().numpy(), axis=0)
    
    probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    ### post_processing of hierarchical violation
    f = open(args.hierarchy_structure)
    taxa = json.load(f)
    taxa = {int(k):int(v) for k,v in taxa.items()}
    probs = violation_process(taxa, probs)

    preds = np.where(probs > THRESHOLD, upper, lower)

    
    return probs

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    ###### 1. training hyperparameters
    parser.add_argument(
        "--model_type",
        type=str,
        default='Linear',
        help="Stacking model type",
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        help="Load trained model",
    )
    parser.add_argument(
        "--host_type",
        type=str,
        help="Host type",
    )
    parser.add_argument(
        "--weighted",
        type=str,
        default='uni',
        help="If passed, add sample wised weight",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=100,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of logging steps.",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=1000, 
        help="set early stop steps"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--train_weight",
        type=str,
        help="train data weight",
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
    
    ###### 3. running mode
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to do prediction on the given dataset.")
    parser.add_argument("--do_predict_no_label", action="store_true", help="Whether to do prediction on novel data.")
    
    
    args = parser.parse_args()
    return args
    
    
def main():
    
    args = parse_args()  
    
    assert args.model_type == 'Linear' or args.model_type == 'MHC' or args.model_type == 'MLP'
    
    if args.seed is not None:
        set_seed(args.seed)
        
    os.chdir(args.base_path)
    
    if args.do_train:
        with open(args.train_label, 'rb') as handle:
            model_label = pickle.load(handle)

        with open(args.train_prob, 'rb') as handle:
            model_prob_agg = pickle.load(handle)
       

        with open(args.valid_label, 'rb') as handle:
            test_label = pickle.load(handle)

        with open(args.valid_prob, 'rb') as handle:
            test_prob_agg = pickle.load(handle)

        with open(args.base_model_list, 'rb') as handle:    
            MODEL_list = pickle.load(handle)
        
        with open(args.LABELS, 'rb') as handle:    
            LABELS = pickle.load(handle)
            
        if args.weighted == 'weighted':
            with open(args.train_weight, 'rb') as handle:
                train_weight = pickle.load(handle)
    
    elif args.do_predict:
        with open(args.valid_label, 'rb') as handle:
            test_label = pickle.load(handle)

        with open(args.valid_prob, 'rb') as handle:
            test_prob_agg = pickle.load(handle)

        with open(args.base_model_list, 'rb') as handle:    
            MODEL_list = pickle.load(handle)
        
        with open(args.LABELS, 'rb') as handle:    
            LABELS = pickle.load(handle)
    
    elif args.do_predict_no_label:

        with open(args.valid_prob, 'rb') as handle:
            test_prob_agg = pickle.load(handle)

        with open(args.base_model_list, 'rb') as handle:    
            MODEL_list = pickle.load(handle)
        
        with open(args.LABELS, 'rb') as handle:    
            LABELS = pickle.load(handle)


    '''Configs'''

    config = configparser.ConfigParser()
    config.num_base= len(MODEL_list)
    if args.host_type == 'Whole':
        config.num_labels = 79
    elif args.host_type == 'Human':
        config.num_labels = 6
    config.feature_size = config.num_labels
    config.feature_dim = config.num_base * config.feature_size
    config.hidden_size = 256
    config.dropout = args.dropout
    config.cls_dropout_prob = args.dropout
    
    MHC_hidden, G2L = 768, 384
    config.g_size = MHC_hidden
    config.G2L = G2L
    config.coe_l = 0
    config.loss_type = 'penalty'

    
    if args.host_type == 'Whole':
        config.hierarchy_classes = [3, 5, 14, 21, 20, 16]     
    
    elif args.host_type == 'Human':
        config.hierarchy_classes = [1, 1, 1, 1, 1, 1]  
        
    config.hierarchical_depth = len(config.hierarchy_classes)



    def jsonKeys2int(x):
        if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
        return x

    fj = open(args.hierarchy_structure)
    hierarchy_structure = json.load(fj)
    hierarchy_structure = jsonKeys2int(hierarchy_structure)
    config.hierarchy_structure = hierarchy_structure


    '''hyper parameter'''
    config.BS = args.batch_size
    config.learning_rate = args.learning_rate
    config.weight_decay = 1e-2
    config.num_train_steps = args.num_train_steps
    
    config.logging_steps = args.logging_steps
    
    if args.model_type == 'Linear':
        model = Linear_Net(config)
        
    elif args.model_type == 'MHC':
        model = MHC_Net(config)
    
    elif args.model_type == 'MLP':
        model = Flat_Net(config)

    
    
    if args.trained_model:
        model.load_state_dict(torch.load(args.trained_model))
    
    '''数据'''
    if args.do_train:
        # 特征需要合并
        train_x, val_x = torch.Tensor(), torch.Tensor()

        # 标签不需要合并
        train_y, val_y = model_label.values, test_label.values
        train_y, val_y = torch.Tensor(train_y), torch.Tensor(val_y)

        # 权重
        if args.weighted == 'weighted':
            train_weight = train_weight.values
            val_weight = np.ones(val_y.shape)
            train_w, val_w = torch.Tensor(train_weight), torch.Tensor(val_weight)
        else:
            train_weight = np.ones(train_y.shape)
            val_weight = np.ones(val_y.shape)
            train_w, val_w = torch.Tensor(train_weight), torch.Tensor(val_weight)
            
        for MODEL in MODEL_list:
            X_train, X_test = model_prob_agg[MODEL].values, test_prob_agg[MODEL].values
            X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)

            # concat base model
            train_x = torch.cat((train_x, X_train), 1)
            val_x = torch.cat((val_x, X_test), 1)


        # Dataset
        
        train_set = TensorDataset(train_x, train_y, train_w)
        val_set = TensorDataset(val_x, val_y, val_w)

        # Dataloader
        train_dataloader = DataLoader(train_set, shuffle=True, drop_last=True, batch_size=config.BS)
        eval_dataloader = DataLoader(val_set, batch_size=config.BS)
    
    
    elif args.do_predict:
         # 特征需要合并
        val_x = torch.Tensor()

        # 标签不需要合并
        val_y = test_label.values
        val_y = torch.Tensor(val_y)

        # 权重
        val_weight = np.ones(val_y.shape)
        val_w =  torch.Tensor(val_weight)
        
        
        for MODEL in MODEL_list:
            X_test = test_prob_agg[MODEL].values
            X_test =  torch.Tensor(X_test)
            val_x = torch.cat((val_x, X_test), 1)

        val_set = TensorDataset(val_x, val_y, val_w)

        # Dataloader
        eval_dataloader = DataLoader(val_set, batch_size=config.BS)
    elif args.do_predict_no_label:
         # 特征需要合并
        val_x = torch.Tensor()
        for MODEL in MODEL_list:
            X_test = test_prob_agg[MODEL].values
            X_test =  torch.Tensor(X_test)
            val_x = torch.cat((val_x, X_test), 1)

        val_set = TensorDataset(val_x)

        # Dataloader
        eval_dataloader = DataLoader(val_set, batch_size=config.BS)
        
            
    
    #MODEL_list = ['bert_G', 'BLAST']

    accelerator = Accelerator(fp16 = True)

    if args.do_train:
        no_decay = ["bias", "LayerNorm.weight"] 
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                    model, optimizer, train_dataloader, eval_dataloader
                )
    else:
        model,  eval_dataloader = accelerator.prepare(
                    model, eval_dataloader)

    
    

    
    
    if args.do_train:
        ### calculate epoch num
        config.num_train_epochs = math.ceil(args.num_train_steps / len(train_dataloader))
    
        save_path = args.save_path + f'{config.BS}_{config.learning_rate}_{config.dropout}'
        Path(save_path).mkdir(parents=True, exist_ok=True)
        os.chdir(save_path)
    
        best_valid_loss = float('inf')
        #best_step, opt_auc, opt_aupr  = 0, 0, 0 
    

        print(f"  Total optimization steps = {config.num_train_steps}")
        print(f"  Batch size = {config.BS}")
        print(f"  Num Epochs = {config.num_train_epochs}")
        
        
        completed_steps = 0
        end_flag = 0
        for epoch in range(config.num_train_epochs):

            start_time = time.monotonic()
            
            
            ###### reset for each epoch
            epoch_train_loss= 0.0
            
            for step, batch in enumerate(train_dataloader):
                model.train()
                # laod and training
                feature=batch[0]
                labels=batch[1]
                weights=batch[2]
                
                outputs = model(feature, labels, weights)
                
                loss = outputs['loss']
                accelerator.backward(loss)
                epoch_train_loss += loss.item()
                
                optimizer.step()
                optimizer.zero_grad()
                completed_steps += 1
                
                
                # logging
                if (completed_steps % config.logging_steps  == 0) & (completed_steps != 0) :
                    eval_loss, result, eval_probs = evaluate(model, eval_dataloader, args)
                    auc = result['auc_micro']
                    aupr = result['aupr_micro']
                    rec = result['recall_mi']
                    prec = result['precision_mi']
                    
                    logging_loss = epoch_train_loss / (step+1)
                    print(f'\t Train Steps: {completed_steps} ')
                    print(f'\t Train Loss: {logging_loss:.3f} ')
                    print(f'\t Val. Loss: {eval_loss:.3f} ')
                    print(f'\t Auc: {auc:.4f} | Aupr: {aupr:.4f}')
                    print(f'\t Rec: {rec:.4f} | Prec: {prec:.4f}')                    
                    
                    ###### early stop  
                    if eval_loss < best_valid_loss:
                        # reset counter
                        stop_counter = 0
                        best_valid_loss = eval_loss
                        best_step = completed_steps
                        opt_auc = auc
                        opt_aupr = aupr
                        torch.save(model.state_dict(), f'best_model.pt')
                        # model_steps_{completed_steps}_{auc:.4f}.pt
                        # only save when better performance achieved
                        pd.DataFrame(eval_probs, columns = LABELS).to_csv(f'best_probs.csv', index = False)
                    else:
                        stop_counter += 1
                    
                    if completed_steps == config.num_train_steps:
                        print(f"Finish training at {completed_steps} steps!")
                        end_flag = 1
                        break
                        
                    elif stop_counter == args.early_stop:
                        print(f"Early stop at {completed_steps} steps!")
                        end_flag = 1
                        break

                        
            end_time = time.monotonic()  
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            
            if end_flag == 1:
                break

        outputFilename = args.save_path + 'results.txt'

        if os.path.exists(outputFilename) == False:
            header = "model\tSteps\tAUC\tAUPR\tLoss\n"
            with open(outputFilename, 'a') as file_writer:
                file_writer.write(header)
                file_writer.close()


        f = open(outputFilename , 'a') 
        f.write(f'Model_{config.BS}_{config.learning_rate}_{config.dropout}:\t')
        f.write(f'{best_step+1}\t')
        f.write(f'{opt_auc:.4f}\t')
        f.write(f'{opt_aupr:.4f}\t')
        f.write(f'{best_valid_loss:.4f}\n')
        f.close()
    
    elif args.do_predict:
        eval_loss, result, eval_probs = predict(model, eval_dataloader, args)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(eval_probs, columns = LABELS).to_csv(args.save_path + 'pred_probs.csv', index=False)
        
    elif args.do_predict_no_label:
        eval_probs = predict_no_label(model, eval_dataloader, args)
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(eval_probs, columns = LABELS).to_csv(args.save_path + 'pred_probs.csv', index=False)
        
        
if __name__ == "__main__":
    main()