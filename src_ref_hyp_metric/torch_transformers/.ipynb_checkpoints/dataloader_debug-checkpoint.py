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
        self.params = params
        self.data = []
        self.label = []
        self.savedata_dir = os.path.join(params.dump_path, '{}.pkl'.format(data_name))
        if not os.path.isfile(self.savedata_dir):
            self.data = self.read_data(self.data_paths, params.forms, tokenizer)
            with open(self.savedata_dir, mode='wb') as w:
                pickle.dump(self.data, w)
        else:
            with open(self.savedata_dir, mode='rb') as r:
                self.data = pickle.load(r)            

    def __len__(self):
        return len(self.data)
    
    ######### needs to be fixed
    def __getitem__(self, idx):
        out_data = self.data[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label 
    
    ### add tokenizing process
    def read_data(self, data_paths, forms, tokenizer):
        """
        data format must be
        
        {data}\t{lang}\n 
        
        
        return data:
        {
         'raw_src':txt, 
         'tok_src':[tok1, tok2, ..., tokn], 
         'raw_ref':txt, 
         'tok_ref':[tok1, tok2, ..., tokn], 
         'raw_hyp':txt, 
         'tok_hyp':[tok1, tok2, ..., tokn], 
         'raw_label':float,
         'tok_label':float
         'lang':language pair
         }
        
        """
        DATA = {form:None for form in forms}
        
        
        for data_path, form in zip(data_paths, forms):
            assert os.path.isfile(data_path)
            with open(data_path, mode='r', encoding='urf-8') as r:
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
                    tmp_dic['raw_{}'.format(form)] = float(d.split('\t')[0])
                    tmp_dic['tok_{}'.format(form)] == tmp_dic['raw_{}'.format(form)]
                else:
                    tmp_dic['raw_{}'.format(form)] = d.split('\t')[0]
                    tmp_dic['tok_{}'.format(form)] = tokenizer.encode(tmp_dic['data'])
                if 'lang' in tmp_dic:
                    assert tmp_dic['lang'] == d.split('\t')[1]
                else:
                    tmp_dic['lang'] = d.split('\t')[1]
            r_data.append(tmp_dic)
        return r_data


# In[ ]:


class Data_Transformer():
    """
    batch : type == list
    batch : 
    [ 
     {
      'raw_src': ~
      'tok_src': ~
      'raw_ref': ~
      ....
      'raw_label': ~
      'tok_label': ~
      'lang':language pair
     }
    ]
    """
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        

