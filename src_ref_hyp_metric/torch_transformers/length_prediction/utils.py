#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
          'xlm-r':[]}


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


# In[7]:


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




