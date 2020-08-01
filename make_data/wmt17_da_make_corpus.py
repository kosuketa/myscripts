#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

DATA_HOME = '/ahc/work3/kosuke-t/data/'

DA_HOME = os.path.join(DATA_HOME, 'WMT/newstest2018-humaneval/analysis')
DARR_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv')
SRC_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/sources')
REF_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/references')
HYP_HOME = os.path.join(DATA_HOME, 'WMT/wmt18-metrics-task-package/source-system-outputs/wmt18-submitted-data/txt/system-outputs/newstest2018')
SAVE_PATH_DARR = os.path.join(DATA_HOME, 'WMT/wmt18_darr.pkl')
SAVE_PATH_DA_GOOD_REDUP = os.path.join(DATA_HOME, 'WMT/wmt18_da_good_redup.pkl')
SAVE_PATH_DA_SEG = os.path.join(DATA_HOME, 'WMT/wmt18_da_seg.pkl')

SAVE_SRC_TRAIN = 'SRHDA/WMT15_18_DA/train.src'
SAVE_REF_TRAIN = 'SRHDA/WMT15_18_DA/train.ref'
SAVE_HYP_TRAIN = 'SRHDA/WMT15_18_DA/train.hyp'
SAVE_LABEL_TRAIN = 'SRHDA/WMT15_18_DA/train.label'
SAVE_SRC_VALID = 'SRHDA/WMT15_18_DA/valid.src'
SAVE_REF_VALID = 'SRHDA/WMT15_18_DA/valid.ref'
SAVE_HYP_VALID = 'SRHDA/WMT15_18_DA/valid.hyp'
SAVE_LABEL_VALID = 'SRHDA/WMT15_18_DA/valid.label'
SAVE_SRC_TEST = 'SRHDA/WMT15_18_DA/test.src'
SAVE_REF_TEST = 'SRHDA/WMT15_18_DA/test.ref'
SAVE_HYP_TEST = 'SRHDA/WMT15_18_DA/test.hyp'
SAVE_LABEL_TEST = 'SRHDA/WMT15_18_DA/test.label'

langs = ['cs-en', 'de-en', 'et-en', 'fi-en', 'ru-en', 'tr-en', 'zh-en', 
         'en-cs', 'en-de', 'en-et', 'en-fi', 'en-ru', 'en-tr', 'en-zh']

# systems = {'cs-en':['CUNI-Transformer.5560', 
#                     'online-A.0', 
#                     'online-B.0', 
#                     'online-G.0', 
#                     'uedin.5561'], 
#            'de-en':[], 
#            'et-en':[], 
#            'fi-en':[], 
#            'ru-en':[], 
#            'tr-en':[], 
#            'zh-en':[], 
#            'en-cs':[], 
#            'en-de':[], 
#            'en-et':[], 
#            'en-fi':[], 
#            'en-ru':[], 
#            'en-tr':[], 
#            'en-zh':[]}

import csv
import pickle
import re
import csv
from pprint import pprint
import pandas as pd
import numpy as np
import copy
from  tqdm import tqdm


# In[2]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(SRC_HOME, 'newstest2018-{0}{1}-src.{0}'.format(lang.split('-')[0], lang.split('-')[1])))  for lang in langs}
REF_files = {lang:load_file(os.path.join(REF_HOME, 'newstest2018-{0}{1}-ref.{1}'.format(lang.split('-')[0], lang.split('-')[1]))) for lang in langs}
HYP_files = {lang:{} for lang in langs}

for lang in langs:
    for fname in os.listdir(os.path.join(HYP_HOME, lang)):
        if not fname.startswith('newstest2018'):
            continue
        # extract system id from fname
        system_id = copy.deepcopy(fname).replace('newstest2018.', '').replace('.{}'.format(lang), '')
        # add
        HYP_files[lang][system_id] = load_file(os.path.join(os.path.join(HYP_HOME, lang), fname))

        


# ↓DARR

# In[3]:


DArr = load_file(DARR_HOME)
corpus = []
for idx, da_data in enumerate(DArr):
    if idx == 0:
        continue
    lang = da_data.split(' ')[0]
    sid = int(da_data.split(' ')[2])
    better_sys = da_data.split(' ')[3]
    worse_sys = da_data.split(' ')[4]
    corpus.append({'lang': lang, 
                   'sid':sid,
                   'src': SRC_files[lang][sid-1], 
                   'ref': REF_files[lang][sid-1], 
                   'hyp1': HYP_files[lang][better_sys][sid-1], 
                   'hyp2': HYP_files[lang][worse_sys][sid-1], 
                   'better':'hyp1'})
print('saving {}'.format(SAVE_PATH_DARR))
with open(SAVE_PATH_DARR, mode='wb') as w:
    pickle.dump(corpus, w)


# DA for train

# In[4]:


filename_good_redup = {lang: os.path.join(DA_HOME, 'ad-{}-good-stnd-redup.csv'.format(lang.replace('-', ''))) for lang in langs}
filename_seg_scores = {lang: os.path.join(DA_HOME, 'ad-seg-scores-{}.csv'.format(lang)) for lang in langs}

DA_data_good_redup = {lang: load_file(f) for lang, f in filename_good_redup.items()}
DA_data_seg_scores = {lang: load_file(f) for lang, f in filename_seg_scores.items()}

def make_corpus_good_stnd_redup(langs, DA_data):
    corpus = []
    type_set = set()
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue

            type_id = row.split('\t')[8]
            score = float(row.split('\t')[-2])
            sid = int(row.split('\t')[9])
            system_id = row.split('\t')[6]

            type_set.add(type_id)

            if type_id != 'SYSTEM':
                continue

            corpus.append({'lang':lang,
                           'sid':sid,
                           'src':SRC_files[lang][sid-1],
                           'ref':REF_files[lang][sid-1],
                           'hyp':HYP_files[lang][system_id][sid-1],
                           'label':score})
    return corpus


def make_corpus_seg_scores(langs, DA_data):
    corpus = []
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue
            system_id = row.split(' ')[0]
            sid = int(row.split(' ')[1])
            score = float(row.split(' ')[3])
            n = int(row.split(' ')[4])
            if system_id == 'HUMAN':
#                 print(score)
                continue
            
            corpus.append({'lang':lang,
                           'sid':sid,
                           'src':SRC_files[lang][sid-1],
                           'ref':REF_files[lang][sid-1],
                           'hyp':HYP_files[lang][system_id][sid-1],
                           'label':score})
    return corpus

corpus_good_redup = make_corpus_good_stnd_redup(langs, DA_data_good_redup)
corpus_seg_scores = make_corpus_seg_scores(langs, DA_data_seg_scores)


# In[5]:


print('good redup')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs}
for corpus in corpus_good_redup:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()

print('seg scores')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()


# In[6]:


print('saving {}'.format(SAVE_PATH_DA_GOOD_REDUP))
with open(SAVE_PATH_DA_GOOD_REDUP, mode='wb') as w:
    pickle.dump(corpus_good_redup, w)
    
print('saving {}'.format(SAVE_PATH_DA_SEG))
with open(SAVE_PATH_DA_SEG, mode='wb') as w:
    pickle.dump(corpus_seg_scores, w)


# In[16]:


SRC_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.src"))
SRC_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.src"))
SRC_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.src"))
REF_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.ref"))
REF_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.ref"))
REF_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.ref"))
HYP_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.hyp"))
HYP_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.hyp"))
HYP_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.hyp"))
LABEL_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.label"))
LABEL_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.label"))
LABEL_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.label"))


# In[17]:


len(SRC_TRAIN17)+len(SRC_VALID17)+len(SRC_TEST17)


# In[14]:





# In[ ]:




