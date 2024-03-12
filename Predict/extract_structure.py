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
from Bio import SeqIO

def parse_args():
    
    parser = argparse.ArgumentParser(description="prepare data")
    parser.add_argument(
        "--predict_folder",
        type=str,
        required=True,
        help="working folder",
        )
    parser.add_argument(
        "--feature_folder",
        type=str,
        required=True,
        help="feature folder",
        )
    parser.add_argument(
        "--assembly_path",
        type=str,
        required=True,
        help="assembly_path to extract metadata",
        )
    parser.add_argument(
        "--struct_evidence",
        type=str,
        required=True,
        help="structure metadata served as evidence",
        )
    parser.add_argument(
        "--struct_evidence_col",
        type=str,
        required=True,
        help="structure metadata served as evidence",
        )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="saved model path",
        )
    parser.add_argument(
        "--categories",
        type=str,
        help="pred class labels"
    )
    parser.add_argument(
        "--feature",
        type=str,
        help="feature model types"
    )
    
    
    args = parser.parse_args()
    
    return args

def add_columns(df, col_list):
    new_col = []
    for col in col_list:
        if col not in df.columns:
            new_col.append(col)
   
    df = df.join(pd.DataFrame(columns = new_col))
    return df[col_list].fillna(0).astype('float')

def np_relu(x):
    return (x > 0) * x

def main():
    args = parse_args()
    
    ####### metadata and dataframe ###########
    header_agg, assembly_agg, ori_agg = [], [], []
    for filename in os.listdir(args.assembly_path):
        header_list = []
        if filename.split('.')[-1] not in ['faa', 'fasta', 'fna']:
            continue
        
        vid = '.'.join(filename.split('.')[:-1])
        with open(args.assembly_path + filename) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                header_list.append(record.id.split("_prot_",1)[1].split('.')[0])
   
        vid_list = [vid] * len(header_list)
        assembly_agg.extend(vid_list)
        header_agg.extend(header_list)
    
    inputs_df = pd.DataFrame({'accession': header_agg,
                      'vid': assembly_agg,
                      'ori_seq': assembly_agg,
                     })
    list_vid_uniq = list(dict.fromkeys(assembly_agg))
    dict_temp = dict(zip(list_vid_uniq, range(len(list_vid_uniq))))
    inputs_df['ori_seq'] = inputs_df['ori_seq'].apply(lambda x: dict_temp[x])
    inputs_df.to_csv(args.predict_folder + 'assembly_df.csv', index=False)
    inputs_df = inputs_df.rename({'accession':'qseqid'}, axis = 1)
    
    ########### extract blast output ###########
    Header = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore' ]
    struct_blast = pd.read_csv(args.feature_folder + 'blast_out.txt', sep='\t', names =Header) ### blast result
    struct_evi_df = pd.read_csv(args.struct_evidence) ### structure evidence
    struct_evi_df = struct_evi_df.rename({'virus_protein': 'sseqid'}, axis = 1)
    struct_cols = pd.read_csv(args.struct_evidence_col)['human_protein'].to_list()

    struct_blast['sseqid'] = struct_blast['sseqid'].apply(lambda x : x.split('|')[1])
    struct_blast['qseqid'] = struct_blast['qseqid'].apply(lambda x : x.split("_prot_",1)[1].split('.')[0])
    struct_blast = struct_blast[['qseqid', 'sseqid', 'pident', 'bitscore']]
    struct_blast = inputs_df.merge(struct_blast,how='left').merge(struct_evi_df,how='left')
    struct_blast = struct_blast[['vid', 'ori_seq','qseqid', 'sseqid', 'pident', 'bitscore', 'human_protein']]
    
    struct_blast.to_csv(args.predict_folder + 'full_struct_evidence.csv', index=False)
    #struct_blast = struct_blast.fillna(0)
    
    ########### aggregate results ###########
    struct_blast = struct_blast[['vid', 'bitscore', 'pident', 'human_protein']].groupby(['vid', 'human_protein']).max('bitscore').reset_index()   

    struct_inter_pid = struct_blast.pivot(index = 'vid', columns = 'human_protein', values='pident')
    struct_inter_pid = add_columns(struct_inter_pid, struct_cols).reset_index()
    
    struct_inter_bit = struct_blast.pivot(index = 'vid', columns = 'human_protein', values='bitscore')#.drop([0], axis=1)
    struct_inter_bit = add_columns(struct_inter_bit, struct_cols).reset_index()
    
    struct_inter = struct_inter_bit.merge(struct_inter_pid, on =['vid'], suffixes=['_bitscore', '_identity'])
    struct_inter.to_csv(args.predict_folder + 'final_struct_evidence.csv', index=False)
    
    ########### prediction according to evidence ###########
    
    with open (args.categories, 'rb') as fp:
        cates = pickle.load(fp)
    
    X_test = struct_inter.iloc[:,1:]
    prob_df, pred_df = pd.DataFrame(), pd.DataFrame()
    
    prob_df['vid'] = struct_inter.iloc[:,0]
    pred_df['vid'] = struct_inter.iloc[:,0]
    p_parent = np.ones(len(X_test))
    
    for COL in cates:  
        with open(args.model_name_or_path + f'{COL}_model.pkl', 'rb') as path:
            clf = pickle.load(path)
        
        p_child = clf.predict_proba(X_test)[:,1]  ### predict
        p_child = p_child - np_relu(p_child - p_parent)
        p_parent = p_child
        preds = np.where(0, 1, p_child > 0.5)
        prob_df[COL] = p_child
        pred_df[COL] = preds
    
    prob_df.to_csv(args.predict_folder + 'whole_probs.csv', index=False)
    pred_df.to_csv(args.predict_folder + 'whole_preds.csv', index=False)
    
    with open(args.predict_folder + "whole_results.txt", "w") as text_file:

        for i in range(len(pred_df)):
            text_file.write(str(pred_df.iloc[i,0]) + '\n')
            text_file.writelines(', '.join(cates[np.where(pred_df.drop(['vid'], axis =1).iloc[i,:] == 1)].to_list())) 
            text_file.write('\n\n')
            
    
if __name__ == "__main__":
    main()