#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

DATA_HOME = '/ahc/work3/kosuke-t/data/'

SAVE_PATH_DA = os.path.join(DATA_HOME, 'WMT/wmt15_da.pkl')
_DATA_HOME = os.path.join(DATA_HOME, 'WMT/DAseg-wmt-newstest2015')


langs = ['cs-en', 'de-en', 'en-ru', 'fi-en', 'ru-en']

import csv
import pickle
import re
import csv
from pprint import pprint
import pandas as pd
import numpy as np
import copy
from  tqdm import tqdm


# In[9]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2015.source.{}'.format(lang)))  for lang in langs}
REF_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2015.reference.{}'.format(lang))) for lang in langs}
HYP_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2015.mt-system.{}'.format(lang))) for lang in langs}
DA_files = {lang:load_file(os.path.join(_DATA_HOME, 'DAseg.newstest2015.human.{}'.format(lang))) for lang in langs}


# In[10]:


corpus = []
for lang in langs:
    for i, (src, ref, hyp, label) in enumerate(zip(SRC_files[lang], REF_files[lang], HYP_files[lang], DA_files[lang])):
        corpus.append({'lang':lang,
                       'sid':int(i)+1,
                       'src':src.rstrip(),
                       'ref':ref.rstrip(),
                       'hyp':hyp.rstrip(),
                       'label':label.rstrip()})
        
        


# In[11]:


print('saving {}'.format(SAVE_PATH_DA))
with open(SAVE_PATH_DA, mode='wb') as w:
    pickle.dump(corpus, w)


# In[ ]:




