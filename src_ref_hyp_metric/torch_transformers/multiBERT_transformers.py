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




