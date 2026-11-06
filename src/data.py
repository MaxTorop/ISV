''' data.py : loads multiple choice MWE style dataset '''

import pandas as pd 
import random
from torch.utils.data import Dataset, DataLoader
import json 
import numpy as np
import re

def format_choices(text):
    text = re.sub(r"\n\s*\(([A-Z])\)", r"\n\1.", text)
    return text

def MWE_CAA(pth =  "./data/myopic-reward.jsonl", tokenizer = None, p_tr = 0.15, p_val = 0.42, batch_train = 20, batch_val = 20, batch_test = 20, seed = 42, model_name = ""):

    ''' prepare MWE style datasets in the MC setting ''' 

    np.random.seed(seed)
    
    try:
        df           = pd.read_json(pth, lines = True) 
    except:
        df           = pd.read_json(pth) 

    df           = df.dropna()

    data_name = pth.split("/")[-1].split(".")[0]
    if data_name == "wealth-seeking-inclination-human":
        df['question'] = df['question'].apply(lambda x : x.split("\n\nAnswer:")[0])
        
    # minor formatting change
    df['question_CAA'] = df["question"].apply(format_choices) + "\n\nRespond with a letter and no other text." 
    try:
        df['answer_matching_behavior'] = df['answer_matching_behavior'].apply(lambda x : x.split( " (")[1].split(")")[0])
        df['answer_not_matching_behavior'] = df['answer_not_matching_behavior'].apply(lambda x : x.split( " (")[1].split(")")[0])
    except:
        df['answer_matching_behavior'] = df['answer_matching_behavior'].apply(lambda x : x.split( "(")[1].split(")")[0])
        df['answer_not_matching_behavior'] = df['answer_not_matching_behavior'].apply(lambda x : x.split( "(")[1].split(")")[0])
    
    N_tr  = int(len(df) * p_tr) # Stratify 
    N_val = int(len(df) * p_val)
    N_te  = len(df) - N_tr - N_val

    df_tr  = df.sample(N_tr, random_state = seed)
    df     = df.drop(df_tr.index)
    df_val = df.sample(N_val, random_state = seed)
    df_te  = df.drop(df_val.index)

    data = {'train' : {'df' : df_tr, 'batch_size' : batch_train},  
            'val' : {'df' : df_val, 'batch_size' : batch_val}, 
            'test' : {'df': df_te, 'batch_size' : batch_test}}

    for split in data.keys():

        df_split      = data[split]['df']
        batch_size    = data[split]['batch_size']
        ordering = [1 if x == "B" else 0 for x in df_split['answer_matching_behavior'].tolist()]

        df_split['ordering']    = ordering 
        data[split]['ordering'] = ordering

        convos_pos = [[{'role' : 'user', 'content' : row['question_CAA']} , {'role' : 'assistant' , 'content' : row['answer_matching_behavior']} ] for idx, row in df_split.iterrows()]
        texts_pos  = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).strip() for convo in convos_pos]

        convos_neg = [[{'role' : 'user', 'content' : row['question_CAA']} , {'role' : 'assistant' , 'content' : row['answer_not_matching_behavior']} ] for idx, row in df_split.iterrows()]
        texts_neg  = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).strip() for convo in convos_neg]

        convos_gen = [[{'role' : 'user', 'content' : row['question_CAA']} ] for idx, row in df_split.iterrows() ]
        texts_gen  = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True) for convo in convos_gen]

        texts_pos = [x[:-1 * len(tokenizer.eos_token)] for x in texts_pos]  # remove eos token for pos/neg so the letter is the final thing
        texts_neg = [x[:-1 * len(tokenizer.eos_token)] for x in texts_neg]
        
        data[split]['pos_templated'] = texts_pos
        data[split]['neg_templated'] = texts_neg
        data[split]['gen_templated'] = texts_gen

        data[split]['df'] = df_split # update

        dataset_pos         = SimpleDataset(texts_pos)
        dataloader_pos      = DataLoader(dataset_pos, batch_size =batch_size, shuffle=False)
        dataset_neg         = SimpleDataset(texts_neg)
        dataloader_neg      = DataLoader(dataset_neg, batch_size = batch_size, shuffle=False)
        dataset_gen         = SimpleDataset(texts_gen)
        dataloader_gen      = DataLoader(dataset_gen, batch_size = batch_size, shuffle = False)

        data[split]['pos_dataloader'] = dataloader_pos
        data[split]['neg_dataloader'] = dataloader_neg
        data[split]['gen_dataloader'] = dataloader_gen

    return data 

class SimpleDataset(Dataset):

    ''' text dataset '''

    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
