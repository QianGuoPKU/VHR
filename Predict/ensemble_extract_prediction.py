import os
import pandas as pd
import argparse
import numpy as np
import random
import csv
import pickle
import re
import torch
import sys
import json


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


def parse_args():
    
    parser = argparse.ArgumentParser(description="prepare data")
    parser.add_argument(
        "--ensemble_path",
        type=str,
        required=True,
        )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        )
    parser.add_argument(
        "--all_cates",
        type=str,
    )
    parser.add_argument(
        "--hierarchy_structure",
        type=str,
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        help="cutoff_path"
    )
    
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    with open (args.all_cates, 'rb') as fp:
        all_cates = pickle.load(fp)
    with open (args.cutoff, 'rb') as fp:
        cutoffs = pickle.load(fp)
        
    f = open(args.hierarchy_structure )
    taxa = json.load(f)
    hierarchy_structure = {int(k):int(v) for k,v in taxa.items()}
    
    '''check folder situations'''
    rec_exist={}
    for mode in ['two_base', 'five_base', 'six_base']:
        rec_exist[mode] = 0
        dirlen = 0 
        for files in os.listdir(args.ensemble_path + f'{mode}'):
            if files.endswith('.pkl'):
                dirlen += 1
        
        if dirlen != 0:
            rec_exist[mode] = 1
            
        
    VIDS, thres, Probs, Preds={}, {}, {}, {}
    
    
    for mode in ['two_base', 'five_base', 'six_base']:
        if rec_exist[mode] == 1:
            '''load vid'''
            with open (args.ensemble_path + f'{mode}/vids.pkl', 'rb') as fp:
                VIDS[mode] = pickle.load(fp)
                
            '''load cutoff'''
            thres[mode] = cutoffs[mode]
            
            '''load probs'''
            Probs[mode] = pd.read_csv(args.ensemble_path + f'{mode}/pred_probs.csv')

    
    
    if rec_exist['five_base'] == 1:
        mode = 'five_base'
        '''special case for six_base'''
        if rec_exist['six_base'] == 1:
            '''update five base using six base'''
            
            '''set index first'''
            Probs['five_base'].insert(0, 'vid', VIDS['five_base'])
            Probs['five_base']=Probs['five_base'].set_index('vid')
            Probs['six_base'].insert(0, 'vid', VIDS['six_base'])
            Probs['six_base']=Probs['six_base'].set_index('vid')
            temp_probs = Probs[mode]
            temp_probs.update(Probs['six_base'])
            
            '''index removed'''
            temp_probs = violation_process(hierarchy_structure, temp_probs.values)
            Probs[mode] = pd.DataFrame(temp_probs, columns = all_cates)
            Preds[mode] = np.where( Probs[mode] > thres['six_base'], 1, 0)
            Preds[mode] = violation_process(hierarchy_structure, Preds[mode])
            
            Preds[mode] = pd.DataFrame(Preds[mode], columns=all_cates)
            Probs[mode].insert(0, 'vid', VIDS[mode])
            Preds[mode].insert(0, 'vid', VIDS[mode])
            Preds[mode].insert(1, 'ensemble', 'nuc_orf')
            Preds[mode].loc[Preds[mode].vid.isin(VIDS['six_base']), 'ensemble'] = 'nuc_orf_struct'
            
        else:
            Preds[mode] = np.where( Probs[mode] > thres[mode], 1, 0)
            Preds[mode] = violation_process(hierarchy_structure, Preds[mode])
            Preds[mode] = pd.DataFrame(Preds[mode], columns=all_cates)
            Probs[mode].insert(0, 'vid', VIDS[mode])
            Preds[mode].insert(0, 'vid', VIDS[mode])
            Preds[mode].insert(1, 'ensemble', 'nuc_orf')

        
    if rec_exist['two_base'] == 1:
        mode = 'two_base'
        Preds[mode] = np.where( Probs[mode] > thres[mode], 1, 0)
        Preds[mode] = violation_process(hierarchy_structure, Preds[mode])
        Preds[mode] = pd.DataFrame(Preds[mode], columns=all_cates)
        Probs[mode].insert(0, 'vid', VIDS[mode])
        Preds[mode].insert(0, 'vid', VIDS[mode])
        Preds[mode].insert(1, 'ensemble', 'nuc')
    
    
    Probs_final, Preds_final = pd.DataFrame(), pd.DataFrame()
    for mode in ['two_base', 'five_base']:
        if rec_exist[mode] == 1:
            Probs_final = Probs_final.append(Probs[mode], ignore_index=True)
            Preds_final = Preds_final.append(Preds[mode], ignore_index=True)
            
    Probs_final= Probs_final.sort_values('vid')
    Preds_final= Preds_final.sort_values('vid')
    
    Probs_final.to_csv(args.output_path + 'whole_probs.csv',index=False,sep=',')
    Preds_final.to_csv(args.output_path + 'whole_preds.csv',index=False,sep=',')
    
    
    with open(args.output_path + "whole_results.txt", "w") as text_file:

        for i in range(len(Preds_final)):
            text_file.write(str(Preds_final.iloc[i,0]) + ': '+str(Preds_final.iloc[i,1]) + '\n')
            text_file.writelines(', '.join(all_cates[np.where(Preds_final.iloc[i,2:] == 1)].to_list())) 
            text_file.write('\n\n')
    
    
    
            
            
if __name__ == "__main__":
    main()