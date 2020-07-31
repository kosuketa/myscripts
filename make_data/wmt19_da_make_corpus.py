#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os

DATA_HOME = '/ahc/work3/kosuke-t/data/'

DA_HOME = os.path.join(DATA_HOME, 'WMT/newstest2019-humaneval')
DARR_HOME = os.path.join(DATA_HOME, 'WMT/wmt19-metrics-task-package/manual-evaluation/RR-seglevel.csv')
SRC_HOME = os.path.join(DATA_HOME, 'WMT/wmt19-metrics-task-package/source-system-outputs/wmt19-submitted-data/txt/sources')
ADDITIONAL_deen_ref = os.path.join(DATA_HOME, 'WMT/wmt19-metrics-task-package/wmt19.newstest2019.HUMAN.de-en')
REF_HOME = os.path.join(DATA_HOME, 'WMT/wmt19-metrics-task-package/source-system-outputs/wmt19-submitted-data/txt/references')
HYP_HOME = os.path.join(DATA_HOME, 'WMT/wmt19-metrics-task-package/source-system-outputs/wmt19-submitted-data/txt/system-outputs/newstest2019')
SAVE_PATH_DARR = os.path.join(DATA_HOME, 'WMT/wmt19_darr.pkl')
# SAVE_PATH_DA_GOOD_REDUP = os.path.join(DATA_HOME, 'WMT/wmt19_da_good_redup.pkl')
# SAVE_PATH_DA_SEG = os.path.join(DATA_HOME, 'WMT/wmt19_da_seg.pkl')

langs = ['cs-de', 'de-cs', 'de-en', 'de-fr', 'en-cs', 'en-de', 'en-fi', 
         'en-gu', 'en-kk', 'en-lt', 'en-ru', 'en-zh', 'fi-en', 'fr-de', 
         'gu-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en']

import csv
import pickle
import re
import csv
from pprint import pprint
import pandas as pd
import numpy as np
import copy
from  tqdm import tqdm


# In[46]:


def load_file(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
        if data[-1] == '':
            data.pop(-1)
    return data

SRC_files = {lang:load_file(os.path.join(SRC_HOME, 'newstest2019-{0}{1}-src.{0}'.format(lang.split('-')[0], lang.split('-')[1])))  for lang in langs}
REF_files = {lang:load_file(os.path.join(REF_HOME, 'newstest2019-{0}{1}-ref.{1}'.format(lang.split('-')[0], lang.split('-')[1]))) for lang in langs}
HYP_files = {lang:{} for lang in langs}

for lang in langs:
    for fname in os.listdir(os.path.join(HYP_HOME, lang)):
        if not fname.startswith('newstest2019'):
            continue
        # extract system id from fname
        system_id = copy.deepcopy(fname).replace('newstest2019.', '').replace('.{}'.format(lang), '')
        # add
        HYP_files[lang][system_id] = load_file(os.path.join(os.path.join(HYP_HOME, lang), fname))

        


# ↓DARR

# In[47]:


DArr = load_file(DARR_HOME)
corpus = []
WrongSysId_dic = {('fi-en', 'Helsinki-NLP.6889'):'Helsinki_NLP.6889', 
                  ('gu-en', 'Ju-Saarland.6525'):'Ju_Saarland.6525', 
                  ('gu-en', 'aylien-mt-gu-en-multilingual.6826'):'aylien_mt_gu-en_multilingual.6826', 
                  ('kk-en', 'DBMS-KU-KKEN.6726'):'DBMS-KU_KKEN.6726', 
                  ('kk-en', 'Frank-s-MT.6127'):'Frank_s_MT.6127',
                  ('kk-en', 'talp-upc-2019-kken.6657'):'talp_upc_2019_kken.6657',
                  ('kk-en', 'rug-kken-morfessor.6677'):'rug_kken_morfessor.6677',
                  ('ru-en', 'Facebook-FAIR.6937'):'Facebook_FAIR.6937', 
                  ('zh-en', 'KSAI-system.6927.zh-en'):'KSAI-system.6927', 
                  ('zh-en', 'online-Y.0.zh-en'):'online-Y.0', 
                  ('zh-en', 'NEU.6832.zh-en'):'NEU.6832', 
                  ('zh-en', 'online-B.0.zh-en'):'online-B.0', 
                  ('zh-en', 'online-A.0.zh-en'):'online-A.0',
                  ('zh-en', 'online-X.0.zh-en'):'online-X.0', 
                  ('zh-en', 'UEDIN.6530.zh-en'):'UEDIN.6530', 
                  ('zh-en', 'online-G.0.zh-en'):'online-G.0', 
                  ('zh-en', 'Baidu-system.6940.zh-en'):'Baidu-system.6940', 
                  ('zh-en', 'BTRANS.6825.zh-en'):'BTRANS.6825', 
                  ('zh-en', 'Apprentice-c.6706.zh-en'):'Apprentice-c.6706', 
                  ('zh-en', 'MSRA.MASS.6942.zh-en'):'MSRA.MASS.6942', 
                  ('zh-en', 'NICT.6814.zh-en'):'NICT.6814', 
                  ('zh-en', 'MSRA.MASS.6996.zh-en'):'MSRA.MASS.6996', 
                  ('zh-en', 'BTRANS-ensemble.6992.zh-en'):'BTRANS-ensemble.6992'}
for idx, da_data in enumerate(DArr):
    if idx == 0:
        continue
    lang = da_data.split(' ')[0]
    sid = int(da_data.split(' ')[2])
    better_sys = da_data.split(' ')[3]
    worse_sys = da_data.split(' ')[4]
    
    check_key_better = (lang, better_sys)
    check_key_worse = (lang, worse_sys)
    if check_key_better in WrongSysId_dic:
        better_sys = WrongSysId_dic[check_key_better]
    if check_key_worse in WrongSysId_dic:
        worse_sys = WrongSysId_dic[check_key_worse]
    
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


# In[48]:


missing_sys


# In[ ]:





# DA for train
# 
# not implemented yet

# In[20]:





# In[16]:


# filename_good_redup = {lang: os.path.join(DA_HOME, 'ad-{}-good-stnd-redup.csv'.format(lang.replace('-', ''))) for lang in langs}
# filename_seg_scores = {lang: os.path.join(DA_HOME, 'ad-seg-scores-{}.csv'.format(lang)) for lang in langs}

# DA_data_good_redup = {lang: load_file(f) for lang, f in filename_good_redup.items()}
# DA_data_seg_scores = {lang: load_file(f) for lang, f in filename_seg_scores.items()}

# def make_corpus_good_stnd_redup(langs, DA_data):
#     corpus = []
#     type_set = set()
#     for lang in langs:
#         for idx, row in enumerate(DA_data[lang]):
#             if idx == 0:
#                 continue

#             type_id = row.split('\t')[8]
#             score = float(row.split('\t')[-2])
#             sid = int(row.split('\t')[9])
#             system_id = row.split('\t')[6]

#             type_set.add(type_id)

#             if type_id != 'SYSTEM':
#                 continue

#             corpus.append({'lang':lang,
#                            'sid':sid,
#                            'src':SRC_files[lang][sid-1],
#                            'ref':REF_files[lang][sid-1],
#                            'hyp':HYP_files[lang][system_id][sid-1],
#                            'label':score})
#     return corpus


# def make_corpus_seg_scores(langs, DA_data):
#     corpus = []
#     for lang in langs:
#         for idx, row in enumerate(DA_data[lang]):
#             if idx == 0:
#                 continue
#             system_id = row.split(' ')[0]
#             sid = int(row.split(' ')[1])
#             score = float(row.split(' ')[3])
#             n = int(row.split(' ')[4])
#             if system_id == 'HUMAN':
# #                 print(score)
#                 continue
            
#             corpus.append({'lang':lang,
#                            'sid':sid,
#                            'src':SRC_files[lang][sid-1],
#                            'ref':REF_files[lang][sid-1],
#                            'hyp':HYP_files[lang][system_id][sid-1],
#                            'label':score})
#     return corpus

# corpus_good_redup = make_corpus_good_stnd_redup(langs, DA_data_good_redup)
# corpus_seg_scores = make_corpus_seg_scores(langs, DA_data_seg_scores)


# In[19]:


# print('good redup')
# print('-- corpus size for each language pair ---')
# lang_count = {lang:0 for lang in langs}
# for corpus in corpus_good_redup:
#     lang = corpus['lang']
#     lang_count[lang] += 1
# for lang in langs:
#     print('{} has {} instances'.format(lang, lang_count[lang]))
# print()

# print('seg scores')
# print('-- corpus size for each language pair ---')
# lang_count = {lang:0 for lang in langs}
# for corpus in corpus_seg_scores:
#     lang = corpus['lang']
#     lang_count[lang] += 1
# for lang in langs:
#     print('{} has {} instances'.format(lang, lang_count[lang]))
# print()


# In[21]:


# print('saving {}'.format(SAVE_PATH_DA_GOOD_REDUP))
# with open(SAVE_PATH_DA_GOOD_REDUP, mode='wb') as w:
#     pickle.dump(corpus_good_redup, w)
    
# print('saving {}'.format(SAVE_PATH_DA_SEG))
# with open(SAVE_PATH_DA_SEG, mode='wb') as w:
#     pickle.dump(corpus_seg_scores, w)


# In[ ]:




