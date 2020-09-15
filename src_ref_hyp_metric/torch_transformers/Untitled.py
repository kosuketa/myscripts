#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os

filepath = '/home/is/kosuke-t/project_disc/data/SRHDA/WMT15_19_DA/test.src'


def load_file(filepath):
    with open(filepath, mode='r', encoding='utf-8') as r:
        data = r.read().split(os.linesep)
    if data[-1] == '':
        data.pop(-1)
    return data

data = load_file(filepath)


# In[ ]:


lang_set = set()
for d in data:
    lang_set.add(d.split('\t')[-1])


# In[ ]:


sorted(list(lang_set))


# In[4]:





# In[8]:





# In[9]:





# In[ ]:




