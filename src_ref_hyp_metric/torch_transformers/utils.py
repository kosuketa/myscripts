#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import pickle
import random
import inspect
import getpass
import argparse
import subprocess
import numpy as np
import torch
from torch import optim
import os
import copy
import time
import json
from collections import OrderedDict

try:
    from transformers import *
except:
    from pytorch_transformers import *
import torch
from torch import nn
import torch.nn.functional as F

from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as sr
import numpy as np
import math
from pprint import pprint

import pickle
import subprocess

import random
import apex
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# In[ ]:


MODELS = {'xlm':['xlm-mlm-en-2048',
                 'xlm-mlm-ende-1024', 
                 'xlm-mlm-enfr-1024',
                 'xlm-mlm-enro-1024',
                 'xlm-mlm-xnli15-1024',
                 'xlm-mlm-tlm-xnli15-1024',
                 'xlm-clm-enfr-1024',
                 'xlm-clm-ende-1024',
                 'xlm-mlm-17-1280',
                 'xlm-mlm-100-1280'], 
          'bert':['bert-base-uncased',
                  'bert-large-uncased',
                  'bert-base-cased',
                  'bert-large-cased',
                  'bert-base-multilingual-uncased',
                  'bert-base-multilingual-cased',
                  'bert-base-chinese',
                  'bert-base-german-cased',
                  'bert-large-uncased-whole-word-masking',
                  'bert-large-cased-whole-word-masking',
                  'bert-large-uncased-whole-word-masking-finetuned-squad',
                  'bert-large-cased-whole-word-masking-finetuned-squad',
                  'bert-base-cased-finetuned-mrpc',
                  'bert-base-german-dbmdz-cased',
                  'bert-base-german-dbmdz-uncased',
                  'bert-base-japanese',
                  'bert-base-japanese-whole-word-masking',
                  'bert-base-japanese-char',
                  'bert-base-japanese-char-whole-word-masking',
                  'bert-base-finnish-cased-v1',
                  'bert-base-finnish-uncased-v1',
                  'bert-base-dutch-cased'], 
          'xlm-r':['xlm-roberta-base', 
                   'xlm-roberta-large', 
                   "xlm-roberta-large-finetuned-conll02-dutch",
                   "xlm-roberta-large-finetuned-conll02-spanish",
                   "xlm-roberta-large-finetuned-conll03-english",
                   "xlm-roberta-large-finetuned-conll03-german"],
         'roberta':['roberta-base', 
                    'roberta-large', 
                    'roberta-large-mnli',
                    'distilroberta-base',
                    'roberta-base-openai-detector',
                    'roberta-large-openai-detector'],
         'reformer':['reformer-enwik8', 
                     'reformer-crime-and-punishment', 
                     'google/reformer-enwik8',
                     'google/reformer-crime-and-punishment']
         }


# In[ ]:


def check_for_langs_id(modelname):
    if modelname in MODELS['xlm']:
        return True
    
    modelnames = []
    for v in MODELS.values():
        modelnames.extend(v)
    assert modelname in modelnames
    
    return False


# In[ ]:


def get_model_type(model_name):
    if model_name in MODELS['bert']:
        return 'bert'
    elif model_name in MODELS['xlm']:
        return 'xlm'
    elif model_name in MODELS['xlm-r']:
        return 'xlm-r'
    elif model_name in MODELS['roberta']:
        return 'roberta'
    elif model_name in MODELS['reformer']:
        return 'reformer'
    else:
        raise NotImplementedError
        exit(-2)


# In[ ]:


def get_tokenizer_class(model_name):
    if model_name in MODELS['bert']:
        return BertTokenizer
    elif model_name in MODELS['xlm']:
        return XlmTokenizer
    elif model_name in MODELS['xlm-r']:
        return XLMRobertaTokenizer
    elif model_name in MODELS['roberta']:
        return RobertaTokenizer
    elif model_name in MODELS['reformer']:
        return ReformerTokenizer
    else:
        raise NotImplementedError
        exit(-2)
        
def get_model_class(model_name):
    if model_name in MODELS['bert']:
        return BertModel
    elif model_name in MODELS['xlm']:
        return XlmModel
    elif model_name in MODELS['xlm-r']:
        return XLMRobertaModel
    elif model_name in MODELS['roberta']:
        return RobertaModel
    elif model_name in MODELS['reformer']:
        return ReformerModel
    else:
        raise NotImplementedError
        exit(-2)
        
def get_config_class(model_name):
    if model_name in MODELS['bert']:
        return BertConfig
    elif model_name in MODELS['xlm']:
        return XlmConfig
    elif model_name in MODELS['xlm-r']:
        return XLMRobertaConfig
    elif model_name in MODELS['roberta']:
        return RobertaConfig
    elif model_name in MODELS['reformer']:
        return ReformerConfig
    else:
        raise NotImplementedError
        exit(-2)


# In[ ]:


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


# In[15]:


def calc_pearson(pred, true):
    try:
        r, p_value = pr(np.asarray(pred), np.asarray(true))
    except ValueError:
        r = -1.0
    return r

def calc_spearman(pred, true):
    try:
        r, p_value = sr(np.asarray(pred), np.asarray(true))
    except ValueError:
        r = -1.0
    return r
def mse(pred, true):
    return (np.square(pred - true)).mean()


# In[ ]:


def reset_gpu(*args):
    for arg in args:
        if arg != None:
            del arg
    torch.cuda.empty_cache()


# In[ ]:


def build_model(args):
    ModelClass = get_model_class(args.model_name)
    ConfigClass = get_config_class(args.model_name)
    
    config = ConfigClass.from_pretrained(args.model_name)
    model = ModelClass.from_pretrained(args.model_name, config=config)
    model.config.num_labels = 1
    if args.hyp_src_hyp_ref:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size*2, 1)])
    else:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size, 1)])
    
    optimizer = get_optimizer(list(model.parameters()), args.optimizer)
    mse = nn.MSELoss()
    
    return model, config, optimizer, mse


# In[9]:


def plot_scatter(results, exp_name, dump_path):
    results = results['test']
    
    X = results['true']
    Y = results['pred']
    figdir1 = os.path.join(dump_path, '{}_scatter.png'.format(exp_name))
    figdir2 = os.path.join(dump_path, '{}_scatter.pdf'.format(exp_name))
    fig = plt.figure(figsize=(7, 5), dpi=100)
    plt.scatter(X, Y)
    plt.xlabel('DA score')
    plt.ylabel('prediction')
    plt.grid()
    plt.savefig(figdir1)
    pp = PdfPages(figdir2)
    pp.savefig(fig)
    pp.close()
    plt.close()


# In[ ]:


def run_meta_evaluation(results):
    pass


# In[35]:


# dump_path = '/ahc/work3/kosuke-t/SRHDA/transformers/log/'
# exp_name = 'multiBERT_all_hyp_src'
# dump_path = os.path.join(os.path.join(dump_path, exp_name), '1')
# with open(os.path.join(dump_path, 'result.pkl'), mode='rb') as f:
#     results = pickle.load(f)
# plot_scatter(results, exp_name, dump_path)

# best_valid_pearson = -1.0
# best_valid_epoch = 0
# for e, v_pearson in enumerate(results['valid']['pearson']):
#     if best_valid_pearson < v_pearson:
#         best_valid_pearson = v_pearson
#         best_valid_epoch = e
# print('--- Final Performance of This Model (Pearson)---')
# print('all : {:.3f}'.format(results['test']['pearson'][best_valid_epoch]))


# In[4]:





# In[5]:





# In[ ]:




