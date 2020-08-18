#!/usr/bin/env python
# coding: utf-8

# In[68]:


import os
import sys

#DATA_HOME = '/ahc/work3/kosuke-t/data/'
DATA_HOME = sys.argv[-1]

DA_HOME = os.path.join(DATA_HOME, 'WMT/DAseg-wmt-newstest2017/ensembled')
HUME_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/manual-evaluation/hume-testset-round-2.tsv')
# DARR_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/manual-evaluation/RR-seglevel.csv')
SRC_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/sources')
REF_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/references')
HYP_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/wmt17-submitted-data/txt/system-outputs/newstest2017')

# SRC_himl_HOME = os.path
# REF_himl_HOME = 
# HYP_himl_a_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/himltest17/txt/system-outputs/himltest2017a')
# HYP_himl_b_HOME = os.path.join(DATA_HOME, 'WMT/wmt17-metrics-task-package/input/wmt17-metrics-task-no-hybrids/himltest17/txt/system-outputshimltest2017b')
# SAVE_PATH_DARR = os.path.join(DATA_HOME, 'WMT/wmt17_darr.pkl')

SAVE_PATH_DA_GOOD_REDUP = os.path.join(DATA_HOME, 'WMT/wmt17_da_good_redup.pkl')
SAVE_PATH_DA_SEG = os.path.join(DATA_HOME, 'WMT/wmt17_da_seg.pkl')

SAVE_SRC_TRAIN = 'SRHDA/WMT15_17_DA_HUME/train.src'
SAVE_REF_TRAIN = 'SRHDA/WMT15_17_DA_HUME/train.ref'
SAVE_HYP_TRAIN = 'SRHDA/WMT15_17_DA_HUME/train.hyp'
SAVE_LABEL_TRAIN = 'SRHDA/WMT15_17_DA_HUME/train.label'
SAVE_SRC_VALID = 'SRHDA/WMT15_17_DA_HUME/valid.src'
SAVE_REF_VALID = 'SRHDA/WMT15_17_DA_HUME/valid.ref'
SAVE_HYP_VALID = 'SRHDA/WMT15_17_DA_HUME/valid.hyp'
SAVE_LABEL_VALID = 'SRHDA/WMT15_17_DA_HUME/valid.label'
SAVE_SRC_TEST = 'SRHDA/WMT15_17_DA_HUME/test.src'
SAVE_REF_TEST = 'SRHDA/WMT15_17_DA_HUME/test.ref'
SAVE_HYP_TEST = 'SRHDA/WMT15_17_DA_HUME/test.hyp'
SAVE_LABEL_TEST = 'SRHDA/WMT15_17_DA_HUME/test.label'

langs_news = ['cs-en', 'de-en', 'en-cs', 'en-de', 'en-fi', 'en-lv', 'en-ru', 
              'en-tr', 'en-zh', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']


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


# In[69]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(SRC_HOME, 'newstest2017-{0}{1}-src.{0}'.format(lang.split('-')[0], lang.split('-')[1])))  for lang in langs_news}
REF_files = {lang:load_file(os.path.join(REF_HOME, 'newstest2017-{0}{1}-ref.{1}'.format(lang.split('-')[0], lang.split('-')[1]))) for lang in langs_news}
HYP_files = {lang:{} for lang in langs_news}

for lang in langs_news:
    for fname in os.listdir(os.path.join(HYP_HOME, lang)):
        if not fname.startswith('newstest2017'):
            continue
        # extract system id from fname
        system_id = copy.deepcopy(fname).replace('newstest2017.', '').replace('.{}'.format(lang), '')
        # add
        HYP_files[lang][system_id] = load_file(os.path.join(os.path.join(HYP_HOME, lang), fname))

        


# ↓DARR

# In[70]:


# DArr = load_file(DARR_HOME)
# corpus = []
# for idx, da_data in enumerate(DArr):
#     if idx == 0:
#         continue
#     lang = da_data.split(' ')[0]
#     sid = int(da_data.split(' ')[2])
#     better_sys = da_data.split(' ')[3]
#     worse_sys = da_data.split(' ')[4]
#     corpus.append({'lang': lang, 
#                    'sid':sid,
#                    'src': SRC_files[lang][sid-1], 
#                    'ref': REF_files[lang][sid-1], 
#                    'hyp1': HYP_files[lang][better_sys][sid-1], 
#                    'hyp2': HYP_files[lang][worse_sys][sid-1], 
#                    'better':'hyp1'})
# print('saving {}'.format(SAVE_PATH_DARR))
# with open(SAVE_PATH_DARR, mode='wb') as w:
#     pickle.dump(corpus, w)


# DA for train

# In[74]:


filename_good_redup = {lang:'' for lang in langs_news}
DA_data_good_redup = {lang:[] for lang in langs_news}
for lang in langs_news:
    file_path = os.path.join(DA_HOME, 'ad-{}-good-stnd.csv'.format(lang.replace('-', '')))
    if os.path.isfile(file_path):
        filename_good_redup[lang] = file_path
        DA_data_good_redup[lang] = load_file(file_path) 
    else:
#         print('{} does not exist'.format(file_path))
        pass
filename_seg_scores = os.path.join(DA_HOME, 'ad-seg-scores-ensembled.csv')
DA_data_seg_scores = load_file(filename_seg_scores)

def make_corpus_good_stnd_redup(langs, DA_data):
    corpus = []
    type_set = set()
    for lang in langs:
        for idx, row in enumerate(DA_data[lang]):
            if idx == 0:
                continue

            type_id = row.split('\t')[8]
            score = float(row.split('\t')[10])
            sid = int(row.split('\t')[9])
            system_id = row.split('\t')[6]

            type_set.add(type_id)

            if type_id != 'SYSTEM':
                continue
            
            if system_id in HYP_files[lang]:
                corpus.append({'lang':lang,
                               'sid':sid,
                               'year':17,
                               'src':SRC_files[lang][sid-1],
                               'ref':REF_files[lang][sid-1],
                               'hyp':HYP_files[lang][system_id][sid-1],
                               'label':score})               
    return corpus, type_set


def make_corpus_seg_scores(DA_data):
    corpus = []
    sys_dic = {}
    for idx, row in enumerate(DA_data):
        if idx == 0:
            continue
        if re.search('SRC TRG HIT N.raw N.z SID SYS RAW.SCR Z.SCR', row):
#             print(idx)
            continue
        lang = '{}-{}'.format(row.split(' ')[0], row.split(' ')[1]) 
        system_id = row.split(' ')[6]
        sid = int(row.split(' ')[5])
        score = float(row.split(' ')[8])
        n = int(row.split(' ')[3])
        key = (lang, system_id)
        if re.search('\+', system_id):
            system_id = system_id.split('+')[0]
            key = (lang, system_id)
            if key not in sys_dic:
                sys_dic[key] = 1
            else:
                sys_dic[key] += 1
            
        corpus.append({'lang':lang,
                       'sid':sid,
                       'year':17,
                       'src':SRC_files[lang][sid-1],
                       'ref':REF_files[lang][sid-1],
                       'hyp':HYP_files[lang][system_id][sid-1],
                       'label':score})
    return corpus, sys_dic

corpus_good_redup, type_set = make_corpus_good_stnd_redup(langs_news, DA_data_good_redup)
corpus_seg_scores, sys_dic = make_corpus_seg_scores(DA_data_seg_scores)


# In[75]:


print('good redup')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs_news}
for corpus in corpus_good_redup:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs_news:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()

print('seg scores')
print('-- corpus size for each language pair ---')
lang_count = {lang:0 for lang in langs_news}
for corpus in corpus_seg_scores:
    lang = corpus['lang']
    lang_count[lang] += 1
for lang in langs_news:
    print('{} has {} instances'.format(lang, lang_count[lang]))
print()


# In[76]:


print('saving {}'.format(SAVE_PATH_DA_GOOD_REDUP))
with open(SAVE_PATH_DA_GOOD_REDUP, mode='wb') as w:
    pickle.dump(corpus_good_redup, w)
    
print('saving {}'.format(SAVE_PATH_DA_SEG))
with open(SAVE_PATH_DA_SEG, mode='wb') as w:
    pickle.dump(corpus_seg_scores, w)


# In[16]:


# SRC_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.src"))
# SRC_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.src"))
# SRC_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.src"))
# REF_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.ref"))
# REF_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.ref"))
# REF_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.ref"))
# HYP_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.hyp"))
# HYP_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.hyp"))
# HYP_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.hyp"))
# LABEL_TRAIN17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/train.label"))
# LABEL_VALID17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/valid.label"))
# LABEL_TEST17 = load_file(os.path.join(DATA_HOME, "SRHDA/WMT15_17_DA/test.label"))


# In[17]:





# In[14]:





# In[ ]:




