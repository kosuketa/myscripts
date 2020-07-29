#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pprint import pprint
import pickle
import argparse
import re
import numpy as np
import pickle
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import copy
import pandas as pd
import difflib
from transformers import *
from pprint import pprint
import numpy as np
import random
import math
from tqdm import tqdm
import torch
from apex import amp
from torch import optim
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from torch import nn

from shutil import rmtree

import logging
random.seed(77)
torch.manual_seed(77)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[ ]:


"""
dataset 
must be build independently from each mode such as 'train', 'dev', 'test:de', test:cs'

preprocessing:
    read data from data_paths
    tokenize data
    make lang_token
    uniform length of mini-batch data and insert pad token
    
"""
class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, tokenizer, data_paths, params, data_name=None):
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_paths = data_paths
        self.args = params
        self.data = []
        self.label = []
        self.savedata_dir = os.path.join(params.dump_path, '{}.pkl'.format(data_name))
        if not os.path.isfile(self.savedata_dir):
            self.data = self.read_data(self.data_paths, tokenizer)
            with open(self.savedata_dir, mode='wb') as w:
                pickle.dump(self.data, w)
        else:
            with open(self.savedata_dir, mode='rb') as r:
                self.data = pickle.load(r)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        out_data = self.data[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data
    
    def get_seqment_id(self, tokens):
        x = []
        x += [0] * len(tokens)
        return x
        
    ### add tokenizing process
    def read_data(self, data_paths, tokenizer):
        forms = ['src', 'label']
        DATA = {form:None for form in forms}
        if tokenizer.bos_token_id != None:
            bos_id = tokenizer.bos_token_id
        else:
            bos_id = tokenizer.cls_token_id
        if tokenizer.eos_token_id != None:
            eos_id = tokenizer.eos_token_id
        else:
            eos_id = tokenizer.sep_token_id
        if tokenizer.sep_token_id != None:
            sep_id = tokenizer.sep_token_id
        else:
            sep_id = tokenizer.eos_token_id
        
        for data_path, form in zip(data_paths, forms):
            assert os.path.isfile(data_path)
            with open(data_path, mode='r', encoding='utf-8') as r:
                data = r.read().split(os.linesep)
                if data[-1] == '':
                    data.pop(-1)
            DATA[form] = data
        r_data = []
        for i in range(len(DATA[forms[0]])):
            tmp_dic = {}
            for form in forms:
                d = DATA[form][i]
                if form == 'label':
                    tmp_dic['{}'.format(form)] = float(d)
                else:
                    tmp_dic['raw_{}'.format(form)] = d
                    
            tmp_dic['tok_src'] = self.tokenizer.encode(tmp_dic['raw_src'])
            tmp_dic['seg_src'] = self.get_seqment_id(tmp_dic['tok_src'])
            r_data.append(tmp_dic)
        
        return r_data


# In[ ]:


class Data_Transformer():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
    
    def __call__(self, batch):
#         import pdb;pdb.set_trace()
        return batch

    def padding(self, tok_list, pad_id=None, lang_padding=False):
        max_seq_len = max([len(x) for x in tok_list])
        bs_size = len(tok_list)
        new_tok_list = []
        for toks in tok_list:
            if pad_id == None:
                if lang_padding:
                    toks += [toks[-1]]*(max_seq_len-len(toks))
                else:
                    toks += [self.pad_id]*(max_seq_len-len(toks))   
            else:
                toks += [pad_id]*(max_seq_len-len(toks))
            new_tok_list.append(toks)
        x = torch.tensor(new_tok_list)
        return x
    
    def collate_fn(self, batch):
        tok_src = []
        seg_src = []
        return_dic = {'raw_src':[], 
                      'label':[]
                     }
        for btch in batch:
            return_dic['raw_src'].append(btch['raw_src'])
            return_dic['label'].append(float(btch['label']))
            tok_src.append(btch['tok_src'])
            seg_src.append(btch['seg_src'])

        return_dic['src'] = self.padding(tok_src)
        return_dic['seg_src'] = self.padding(seg_src, pad_id=0)
        
        return_dic['label'] = torch.FloatTensor(return_dic['label'])
    
        return return_dic


# In[ ]:





# In[7]:


# import torch

# a = torch.ones(1, 5)
# b = torch.ones(1, 8)
# print(a)
# print(b)

# torch.nn.utils.rnn.pad_sequence([b, a])


# In[ ]:




