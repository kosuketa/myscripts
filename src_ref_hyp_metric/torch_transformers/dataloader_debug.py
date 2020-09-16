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
from torch import optim
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from torch import nn

from shutil import rmtree

import logging
import utils
random.seed(77)
# torch.manual_seed(77)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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
    def __init__(self, transform, tokenizer, data_paths, args, data_name=None, test=False):
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_paths = data_paths
        self.args = args
        self.test = test
        self.data = []
        self.label = []
        self.savedata_dir = os.path.join(args.tmp_path, '{}.pkl'.format(data_name))
        if not os.path.isfile(self.savedata_dir):
            self.data = self.read_data(self.data_paths, tokenizer)
            with open(self.savedata_dir, mode='wb') as w:
                pickle.dump(self.data, w)
        else:
            with open(self.savedata_dir, mode='rb') as r:
                self.data = pickle.load(r)
                
        self.limit_lang()
        if self.args.train_shrink < 1.0:
            self.data = random.sample(self.data, int(len(self.data)/2))
    
    def limit_lang(self):
        data_list = []
        for data in self.data:
            if data['lang'] in self.args.langs:
                data_list.append(data)
        self.data = data_list

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        out_data = self.data[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data
    
    def encode2sents(self, sent1, sent2, tokenizer):
        return tokenizer.encode(sent1, sent2)
        
    def encode3sents(self, sent1, sent2, sent3, tokenizer, bos_id, sep_id, eos_id, insert_bos=False):
        x = tokenizer.encode(sent1, sent2)
        if not insert_bos:
            x[-1] = sep_id
        else:
            x.append(bos_id)
        sent3_ids = tokenizer.encode(sent3)
        sent3_ids.pop(0)
        x.extend(sent3_ids)
        return x
    
    def get_seqment_id(self, tokens, bos_id, sep_id, eos_id):
        x = []
        index = 0
        for idx, tok in enumerate(tokens):
            if tok in [sep_id, eos_id]:
                index = idx
                break
        x += [0] * (index+1)
        x += [1] * (len(tokens)-len(x))
        return x
    
    def get_lang_id(self, lang_pair, tokens, sep_id, eos_id, use_src=True, hyp_src_ref=False):
        lang1_id = self.tokenizer.lang2id[lang_pair.split('-')[0]]
        lang2_id = self.tokenizer.lang2id[lang_pair.split('-')[1]]
        x = []
        index = 0
        index2 = 0
        for idx, tok in enumerate(tokens):
            if tok in [sep_id, eos_id] and idx != 0:
                if index == 0:
                    index = idx
                else:
                    index2 = idx
                if (not hyp_src_ref) or index2 != 0:
                    break
                
        x += [lang2_id] * (index+1)
        if not hyp_src_ref:
            if use_src:
                x += [lang1_id] * (len(tokens)-len(x))
            else:
                x += [lang2_id] * (len(tokens)-len(x))
        
        else:
            x += [lang1_id] * (index2-(index+1))
            x += [lang2_id] * (len(tokens)-len(x))
        
        return x
        
    ### add tokenizing process
    def read_data(self, data_paths, tokenizer):
        """
        data format must be
        
        {data}\t{lang}\n 
        
        
        return data:
        {
         'raw_src':txt, 
         'raw_ref':txt, 
         'raw_hyp':txt, 
         'label':float,
         'lang':language pair
         'tok_hyp_src': ,
         'tok_hyp_src_ref',
         'tok_hyp_ref':
         'seg_hyp_src':,
         'seg_hyp_src_ref',
         'seg_hyp_ref':
         }
        
        """
        forms = ['src', 'ref', 'hyp', 'label']
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
            if DATA[forms[0]][i].split('\t')[1] not in self.args.langs:
                continue
            for form in forms:
                d = DATA[form][i]
                lang = d.split('\t')[1]
                if form == 'label':
                    if self.test and self.args.darr:
                        tmp_dic[form] = str(d.split('\t')[0])
                    else:
                        tmp_dic[form] = float(d.split('\t')[0])
                else:
                    tmp_dic['raw_{}'.format(form)] = d.split('\t')[0]
                if 'lang' in tmp_dic and form != 'label':
#                     if not tmp_dic['lang'] == d.split('\t')[1]:
#                         import pdb;pdb.set_trace()
                    assert tmp_dic['lang'] == lang
                else:
                    tmp_dic['lang'] = lang
            if 'src' in self.args.forms:
                tmp_dic['tok_hyp_src'] = self.encode2sents(tmp_dic['raw_hyp'], tmp_dic['raw_src'], tokenizer)
                if len(tmp_dic['tok_hyp_src']) >= tokenizer.model_max_length:
                    tmp_dic['tok_hyp_src'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.85)], tmp_dic['raw_src'], tokenizer)
                    if len(tmp_dic['tok_hyp_src']) >= tokenizer.model_max_length:
                        tmp_dic['tok_hyp_src'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.75)], tmp_dic['raw_src'], tokenizer)
                        if len(tmp_dic['tok_hyp_src']) >= tokenizer.model_max_length:
                            tmp_dic['tok_hyp_src'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.65)], tmp_dic['raw_src'], tokenizer)
                    if len(tmp_dic['tok_hyp_src']) >= tokenizer.model_max_length:
                        self.args.logger.error('seqence token length ({}) is over model_max_length ({})'.format(len(tmp_dic['tok_hyp_src']), tokenizer.model_max_length))
                        #import pdb;pdb.set_trace()
                tmp_dic['seg_hyp_src'] = self.get_seqment_id(tmp_dic['tok_hyp_src'], bos_id, sep_id, eos_id)
                if self.args.lang_id_bool:
                    tmp_dic['lang_hyp_src'] = self.get_lang_id(tmp_dic['lang'], tmp_dic['tok_hyp_src'], sep_id, eos_id)
                
                if 'ref' in self.args.forms:
                    tmp_dic['tok_hyp_src_ref'] = self.encode3sents(tmp_dic['raw_hyp'], 
                                                                   tmp_dic['raw_src'], 
                                                                   tmp_dic['raw_ref'], 
                                                                   tokenizer, bos_id, sep_id, eos_id)
                    if len(tmp_dic['tok_hyp_src_ref']) > tokenizer.model_max_length:
                        tmp_dic['tok_hyp_src_ref'] = self.encode3sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.85)], 
                                                                   tmp_dic['raw_src'][:int(len(tmp_dic['raw_src'])*0.85)], 
                                                                   tmp_dic['raw_ref'][:int(len(tmp_dic['raw_ref'])*0.85)], 
                                                                   tokenizer, bos_id, sep_id, eos_id)
                        if len(tmp_dic['tok_hyp_src_ref']) > tokenizer.model_max_length:
                            tmp_dic['tok_hyp_src_ref'] = self.encode3sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.70)], 
                                                                       tmp_dic['raw_src'][:int(len(tmp_dic['raw_src'])*0.70)], 
                                                                       tmp_dic['raw_ref'][:int(len(tmp_dic['raw_ref'])*0.70)], 
                                                                       tokenizer, bos_id, sep_id, eos_id)
                            if len(tmp_dic['tok_hyp_src_ref']) > tokenizer.model_max_length:
                                tmp_dic['tok_hyp_src_ref'] = self.encode3sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.50)], 
                                                                           tmp_dic['raw_src'][:int(len(tmp_dic['raw_src'])*0.50)], 
                                                                           tmp_dic['raw_ref'][:int(len(tmp_dic['raw_ref'])*0.50)], 
                                                                           tokenizer, bos_id, sep_id, eos_id)
                    tmp_dic['seg_hyp_src_ref'] = self.get_seqment_id(tmp_dic['tok_hyp_src_ref'], bos_id, sep_id, eos_id)
                    if self.args.lang_id_bool:
                        tmp_dic['lang_hyp_src_ref'] = self.get_lang_id(tmp_dic['lang'], tmp_dic['tok_hyp_src_ref'], sep_id, eos_id, hyp_src_ref=True)
            if 'ref' in self.args.forms:
                tmp_dic['tok_hyp_ref'] = self.encode2sents(tmp_dic['raw_hyp'], tmp_dic['raw_ref'], tokenizer)
                if len(tmp_dic['tok_hyp_ref']) >= tokenizer.model_max_length:
                    tmp_dic['tok_hyp_ref'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.85)], tmp_dic['raw_ref'], tokenizer)
                    if len(tmp_dic['tok_hyp_ref']) >= tokenizer.model_max_length:
                        tmp_dic['tok_hyp_ref'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.75)], tmp_dic['raw_ref'], tokenizer)
                        if len(tmp_dic['tok_hyp_ref']) >= tokenizer.model_max_length:
                            tmp_dic['tok_hyp_ref'] = self.encode2sents(tmp_dic['raw_hyp'][:int(len(tmp_dic['raw_hyp'])*0.65)], tmp_dic['raw_ref'], tokenizer)
                    if len(tmp_dic['tok_hyp_ref']) >= tokenizer.model_max_length:
                        self.args.logger.error('seqence token length ({}) is over model_max_length ({})'.format(len(tmp_dic['tok_hyp_ref']), tokenizer.model_max_length))
                        #import pdb;pdb.set_trace()
                tmp_dic['seg_hyp_ref'] = self.get_seqment_id(tmp_dic['tok_hyp_ref'], bos_id, sep_id, eos_id)
                if self.args.lang_id_bool:
                    tmp_dic['lang_hyp_ref'] = self.get_lang_id(tmp_dic['lang'], tmp_dic['tok_hyp_ref'], sep_id, eos_id, use_src=False)
                    
            r_data.append(tmp_dic)
        return r_data


# In[ ]:


class Data_Transformer():
    """
    batch : type == dict
    batch : 
    {
      'raw_src': [~]
      'tok_src': [~]
      'raw_ref': [~]
      ....
      'raw_label': [~]
      'tok_label': [~]
      'lang':language pair
    }
    
    """
    
    def __init__(self, args, tokenizer, test=False):
        self.args = args
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.test = test
    
    def __call__(self, batch):
#         import pdb;pdb.set_trace()
        return batch


    def padding(self, tok_list, pad_id=None, lang_padding=False):
        args = self.args
        if utils.get_model_type(args.model_name) == 'reformer':
            max_seq_len = max([len(x) for x in tok_list])
            if not max_seq_len % args.model_config.lsh_attn_chunk_length == 0:
                max_seq_len = (int(max_seq_len / args.model_config.lsh_attn_chunk_length) + 1) *  args.model_config.lsh_attn_chunk_length
        else:
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
        tok_hyp_src = []
        tok_hyp_ref = []
        tok_hyp_src_ref = []
        seg_hyp_src = []
        seg_hyp_ref = []
        seg_hyp_src_ref = []
        lang_hyp_src = []
        lang_hyp_ref = []
        lang_hyp_src_ref = []
        return_dic = {'raw_src':[], 
                      'raw_ref':[], 
                      'raw_hyp':[], 
                      'label':[], 
                      'lang':[]
                     }
        for btch in batch:
            return_dic['raw_src'].append(btch['raw_src'])
            return_dic['raw_ref'].append(btch['raw_ref'])
            return_dic['raw_hyp'].append(btch['raw_hyp'])
            if self.args.darr and self.test:
                return_dic['label'].append(btch['label'])
            else:
                return_dic['label'].append(float(btch['label']))
            if 'src' in self.args.forms:
                tok_hyp_src.append(btch['tok_hyp_src'])
                seg_hyp_src.append(btch['seg_hyp_src'])
                if self.args.lang_id_bool:
                    lang_hyp_src.append(btch['lang_hyp_src'])
                if 'ref' in self.args.forms:
                    tok_hyp_src_ref.append(btch['tok_hyp_src_ref'])
                    seg_hyp_src_ref.append(btch['seg_hyp_src_ref'])
                    if self.args.lang_id_bool:
                        lang_hyp_src_ref.append(btch['lang_hyp_src_ref'])
            if 'ref' in self.args.forms:
                tok_hyp_ref.append(btch['tok_hyp_ref'])
                seg_hyp_ref.append(btch['seg_hyp_ref'])
                if self.args.lang_id_bool:
                    lang_hyp_ref.append(btch['lang_hyp_ref'])
            return_dic['lang'].append(btch['lang'])
        
        if 'src' in self.args.forms:
            return_dic['hyp_src'] = self.padding(tok_hyp_src)
            return_dic['seg_hyp_src'] = self.padding(seg_hyp_src, pad_id=1)
            if self.args.lang_id_bool:
                return_dic['lang_hyp_src'] = self.padding(lang_hyp_src, lang_padding=True)
            if 'ref' in self.args.forms:
                return_dic['hyp_src_ref'] = self.padding(tok_hyp_src_ref)
                return_dic['seg_hyp_src_ref'] = self.padding(seg_hyp_src_ref, pad_id=1)
                if self.args.lang_id_bool:
                    return_dic['lang_hyp_src_ref'] = self.padding(lang_hyp_src_ref, lang_padding=True)
        if 'ref' in self.args.forms:
            return_dic['hyp_ref'] = self.padding(tok_hyp_ref)
            return_dic['seg_hyp_ref'] = self.padding(seg_hyp_ref, pad_id=1)
            if self.args.lang_id_bool:
                return_dic['lang_hyp_ref'] = self.padding(lang_hyp_ref, lang_padding=True)
        
        if self.args.darr and self.test:
            return_dic['label'] = torch.FloatTensor([0.0] * len(return_dic['label']))
        else:
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


# In[29]:


# from transformers import XLMRobertaModel, XLMRobertaTokenizer
# import torch

# model_name = 'xlm-roberta-large'
# tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
# model = XLMRobertaModel.from_pretrained(model_name)

# input_ids = torch.load('/ahc/work3/kosuke-t/SRHDA/transformers/log/xlm-r-large_hyp_src_ref/1/debug_data.pth')  # Batch size 1
# outputs = model(input_ids)
# sentvec = outputs[1]


# In[28]:


# input_ids.shape


# In[30]:





# In[15]:





# In[ ]:




