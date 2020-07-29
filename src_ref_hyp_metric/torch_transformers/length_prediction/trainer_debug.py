#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import utils
from shutil import rmtree
import apex
from torch import nn
import torch.nn.functional as F

import logging
random.seed(77)
torch.manual_seed(77)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataloader_debug import Dataset, Data_Transformer


# In[ ]:


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

# return True when only 1 item is True and others are all False
def bool_check_onlyone_stands(bools):
    length = len(bools)
    length_ls = list(range(length))
    for i in range(length):
        if bools[i]:
            copy_length_ls = copy.deepcopy(length_ls)
            copy_length_ls.pop(i)
            ex_is = copy_length_ls
            for e_i in ex_is:
                if bools[e_i]:
                    return False
            return True
        else:
            continue
    return False


# In[ ]:


parser = argparse.ArgumentParser()

# general setting
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--exp_id', type=str, default='1')
parser.add_argument('--dump_path', type=str, default='/ahc/work3/kosuke-t/SRHDA/XLM/log/')
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--empty_dump', type=bool_flag, default=False)
parser.add_argument('--train', type=bool_flag, default=True)
parser.add_argument('--test', type=bool_flag, default=True)

# hyperparameters
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch_size', type=int, default=3)
parser.add_argument('--optimizer', type=str, default='adam,lr=0.00001')
parser.add_argument('--lr_lambda', type=float, default=0.707, help='lr decay rate')
parser.add_argument('--dropout', type=float, default=0.0, help='mlp dropout')

# model setting
parser.add_argument('--amp', type=bool_flag, default=False)
parser.add_argument('--load_model', type=bool_flag, default=False)
parser.add_argument('--load_model_path', type=str, default='')
parser.add_argument('--save_model_name', type=str, default='model.pth')
parser.add_argument('--save_model_path', type=str, default='')

# data setting
parser.add_argument('--src_train', type=str, default='', help='src data path for train')
parser.add_argument('--src_valid', type=str, default='', help='src data path for train')
parser.add_argument('--src_test', type=str, default='', help='src data path for train')
parser.add_argument('--label_train', type=str, default='', help='label data path for train')
parser.add_argument('--label_valid', type=str, default='', help='label data path for train')
parser.add_argument('--label_test', type=str, default='', help='label data path for train')

parser.add_argument('--train_shrink', type=float, default=1.0)


args = parser.parse_args()

# make dump_path
args.dump_path = os.path.join(os.path.join(args.dump_path, args.exp_name), args.exp_id)
if not os.path.isdir(args.dump_path):
    os.makedirs(args.dump_path)
elif args.empty_dump:
    rmtree(args.dump_path)
    os.makedirs(args.dump_path)
if args.save_model_path == '':
    args.save_model_path = os.path.join(args.dump_path, args.save_model_name)

args.data_paths_train = [args.src_train, args.label_train]
args.data_paths_valid = [args.src_valid, args.label_valid]
args.data_paths_test = [args.src_test, args.label_test]
        
if not (args.train or args.test):
    print('ERROR: argument -train or -test, either of them must be true!')
    exit(-2)
    
logging.basicConfig(filename=os.path.join(args.dump_path, 'logger.log'), level=logging.INFO)
args.logger = logging
data_paths = []


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_trans = Data_Transformer(args, tokenizer)
DATA = {}
if args.train:
    DATA['train'] = Dataset(data_trans, tokenizer, args.data_paths_train, args, '{}.train'.format(args.exp_name))
    DATA['valid'] = Dataset(data_trans, tokenizer, args.data_paths_valid, args, '{}.valid'.format(args.exp_name))
if args.test:
    DATA['test'] = Dataset(data_trans, tokenizer, args.data_paths_test, args, '{}.test'.format(args.exp_name))


# In[ ]:


def update_lr(optimizer, args):
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_lambda
    return optimizer

def run_model(model, batch, args, loss_fn, optimizer, train=False):
    x = batch['src'].to('cuda')
    seg_x = batch['seg_src'].to('cuda')
    labels = batch['label'].to('cuda')
    
    h = model(x, token_type_ids=seg_x)[1]
    preds = model.mlp(h)
    loss = loss_fn(preds.view(-1), labels.view(-1))
    
    preds = [float(p) for p in preds.view(-1).cpu().detach().numpy()]
    labels = [float(t) for t in labels.view(-1).cpu().detach().numpy()]
    
    if train:
        if not args.amp:
            loss.backward()
            optimizer.step()
        else:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
    
    return loss.item(), preds, labels


# In[17]:


# import pdb;pdb.set_trace()

train_dataloader = torch.utils.data.DataLoader(DATA['train'], 
                                               batch_size=args.batch_size, 
                                               collate_fn=data_trans.collate_fn, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(DATA['valid'], batch_size=args.batch_size,
                                               collate_fn=data_trans.collate_fn, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(DATA['test'], batch_size=args.batch_size,
                                              collate_fn=data_trans.collate_fn, shuffle=False)

model = AutoModel.from_pretrained(args.model_name)
model.config.num_labels = 1
model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size, 1)]).cuda()
model.to('cuda')
optimizer = utils.get_optimizer(list(model.parameters()), args.optimizer)
mse = nn.MSELoss()

if args.amp:
    model, optimizer = apex.amp.initialize(
        model,
        optimizer,
        opt_level=('O%i' % 1)
    )

results = {mode:{key:[] for key in ['loss', 'pred', 'true']} for mode in ['train', 'valid', 'test']}
best_val_loss = 10000
for n_epoch in range(args.epoch_size):
    # train
    if args.train:
        model.train()
        losses = []
        preds_ls = []
        trues_ls = []
        for batch_data in train_dataloader:
            optimizer.zero_grad()
            loss, preds, labels = run_model(model, batch_data, args, mse, optimizer, train=True)
            losses.append(loss)
            preds_ls.extend(preds)
            trues_ls.extend(labels)
        
        results['train']['loss'].append(np.mean(losses))
        results['train']['pred'].append(preds_ls)
        results['train']['true'].append(trues_ls)
            
        # valid
        model.eval()
        losses = []
        preds_ls = []
        trues_ls = []
        for batch_data in valid_dataloader:
            with torch.no_grad():
                loss, preds, labels = run_model(model, batch_data, args, mse, optimizer)
            losses.append(loss)
            preds_ls.extend(preds)
            trues_ls.extend(labels)
            
        results['valid']['loss'].append(np.mean(losses))
        results['valid']['pred'].append(preds_ls)
        results['valid']['true'].append(trues_ls)
        
        # update lr
        if best_val_loss > np.mean(losses):
            best_val_loss = np.mean(losses)
        else:
            optimizer = update_lr(optimizer, args)
        
    # test
    if args.test:
        model.eval()
        losses = []
        preds_ls = []
        trues_ls = []
        for batch_data in test_dataloader:
            with torch.no_grad():
                loss, preds, labels = run_model(model, batch_data, args, mse, optimizer)
            losses.append(loss)
            preds_ls.extend(preds)
            trues_ls.extend(labels)
            
        results['test']['loss'].append(np.mean(losses))
        results['test']['pred'].append(preds_ls)
        results['test']['true'].append(trues_ls)   
    
    print('-----------------')
    print('{}epoch finished!'.format(n_epoch+1))
    print('lr = {}'.format(optimizer.param_groups[0]['lr']))
    if args.train:
        print('train loss_mean:{:.4f}'.format(results['train']['loss'][-1]))
        print('valid loss_mean:{:.4f}'.format(results['valid']['loss'][-1]))
    if args.test:
        print('test loss_mean:{:.4f}'.format(results['test']['loss'][-1]))
        
    print('-----------------')
    
with open(os.path.join(args.dump_path, 'result.pkl'), mode='wb') as w:
    pickle.dump(results, w)
    


# In[ ]:


# bash run.sh --exp_name multiBERT_all_hyp_ref --optimizer adam,lr=0.00003 --batch_size 16 --epoch_size 20 --hyp_src True --hyp_src_hyp_ref False

