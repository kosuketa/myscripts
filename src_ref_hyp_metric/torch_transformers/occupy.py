#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# torch.autograd.set_detect_anomaly(True)


# In[ ]:


class ARGS():
    def __init__(self):
        self.data_home = '/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA'
        self.exp_name = 'multi-BERT_15'
        self.dump_path = '/ahc/work3/kosuke-t/SRHDA/transformers/log/'
        self.data_dirs = {'src_train':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.src', 
                          'src_valid':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.src', 
                          'src_test':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.src', 
                          'ref_train':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.ref', 
                          'ref_valid':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.ref', 
                          'ref_test':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.ref', 
                          'hyp_train':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.hyp', 
                          'hyp_valid':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.hyp', 
                          'hyp_test':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.hyp', 
                          'label_train':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/train.label', 
                          'label_valid':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/valid.label', 
                          'label_test':'/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/test.label'
                         }
    
        self.dump_path = os.path.join(self.dump_path, self.exp_name)
        if not os.path.isdir(args.dump_path):
            os.makedirs(args.dump_path)
    
    


# In[ ]:


from transformers import BertModel, BertTokenizer
import torch

string = "Hello, my dog is cute" * 50

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.cuda()
input_ids = torch.tensor([[[tokenizer.encode(string, add_special_tokens=True)*1000]*100]*50])  # Batch size 1
input_ids.cuda()
import pdb;pdb.set_trace()
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple


# In[13]:


# import torch
# x = torch.ones(3,3)
# print(x)
# x_list = [2,2]
# x[0,] = torch.FloatTensor(x_list)
# print(x)


# In[21]:


import os
import pickle
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import numpy as np

def calc_pearson(pred, true):
    try:
        r, p_value = pr(np.asarray(pred), np.asarray(true))
    except ValueError:
        r = -1.0
    return r

EXP_NAMES = ['multiBERT_all_hyp_src_hyp_ref', 
             'multiBERT_all_hyp_src_ref', 
             'multiBERT_all_hyp_src',
             'multiBERT_all_hyp_ref',
             'multiBERT_15_hyp_src_hyp_ref', 
             'multiBERT_15_hyp_src_ref', 
             'multiBERT_15_hyp_src',
             'multiBERT_15_hyp_ref',
             'multiBERT_halved_hyp_src_hyp_ref', 
             'multiBERT_halved_hyp_src_ref', 
             'multiBERT_halved_hyp_src',
             'multiBERT_halved_hyp_ref',]

datadir = '/ahc/work3/kosuke-t/SRHDA/transformers/log/'
for exp_name in EXP_NAMES:
    data_path = os.path.join(os.path.join(datadir, exp_name), '1')
    result_file = os.path.join(data_path, 'result.pkl')
    with open(result_file, mode='rb') as r:
        results = pickle.load(r)
    best_val_epoch = 0
    best_val_pearson = 0

    for e, p_val in enumerate(results['valid']['pearson']):
        if best_val_pearson < p_val:
            best_val_pearson = p_val
            best_val_epoch = e

    highs = {'pred':[], 'true':[]}
    lows = {'pred':[], 'true':[]}
    for pred, true in zip(results['test']['pred'][best_val_epoch], results['test']['true'][best_val_epoch]):
        if true >= 0.0:
            highs['pred'].append(pred)
            highs['true'].append(true)
        else:
            lows['pred'].append(pred)
            lows['true'].append(true)

    print('-----------')
    print(exp_name)
    print('All\tDA >= 0.0\tDA<0.0\tRD')
    print('{:.3f}\t{:.3f}.\t{:.3}\t{:.2f}'.format(results['test']['pearson'][best_val_epoch], 
                                  calc_pearson(highs['pred'], highs['true']), 
                                  calc_pearson(lows['pred'], lows['true']), 
                                  (calc_pearson(highs['pred'], highs['true'])-calc_pearson(lows['pred'], lows['true']))*100/calc_pearson(highs['pred'], highs['true'])
                                 )
         )


# In[18]:





# In[ ]:




