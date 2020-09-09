#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pickle
import random


# In[3]:


SAVE_PATH = '/ahc/work3/kosuke-t/data/SRHDA/WMT15_17_DA/'

def load_file(fname):
    file_ls = []
    with open(fname, mode='r', encoding='utf-8') as r:
        file_ls = r.read().split(os.linesep)
        if file_ls[-1] == "":
            file_ls.pop(-1)
    return file_ls


# In[4]:


src = load_file(os.path.join(SAVE_PATH, 'test.src'))
ref = load_file(os.path.join(SAVE_PATH, 'test.ref'))
hyp = load_file(os.path.join(SAVE_PATH, 'test.hyp'))
label = load_file(os.path.join(SAVE_PATH, 'test.label'))


# In[8]:


idx = random.choice(list(range(len(hyp))))

print('src')
print(src[idx])
print()
print('ref')
print(ref[idx])
print()
print('hyp')
print(hyp[idx])
print()
print('label')
print(label[idx])


# In[9]:


del model


# In[ ]:




