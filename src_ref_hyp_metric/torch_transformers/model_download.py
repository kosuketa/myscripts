#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch

import utils
import sys

# model_name = str(sys.argv[-2])
# SAVE_DIR = str(sys.argv[-1])
MODELS = utils.MODELS

# model_name = 'xlm-roberta-large'
#DEST_DIR = '/home/is/kosuke-t/tmp'
DEST_DIR = sys.argv[-1]


# In[2]:


for key in MODELS.keys():
    for model_name in MODELS[key]: 
        SAVE_DIR = os.path.join(DEST_DIR, model_name)
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        TokenizerClass = utils.get_tokenizer_class(model_name)
        ModelClass = utils.get_model_class(model_name)
        ConfigClass = utils.get_config_class(model_name)

        tokenizer = TokenizerClass.from_pretrained(model_name)
        config = ConfigClass.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name, config=config)

        tokenizer.save_pretrained(SAVE_DIR)
        model.save_pretrained(SAVE_DIR)


# In[4]:


# tokenizer = TokenizerClass.from_pretrained(SAVE_DIR)
# model = ModelClass.from_pretrained(SAVE_DIR)


# In[ ]:




