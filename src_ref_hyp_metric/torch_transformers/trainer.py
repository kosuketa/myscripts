#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from pprint import pprint
import numpy as np
import random
import math
from tqdm import tqdm
import torch
from torch import optim
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import utils
from shutil import rmtree
import shutil
import apex
from torch import nn
import torch.nn.functional as F
import time

random.seed(77)
# torch.manual_seed(77)
# np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

from dataloader_debug import Dataset, Data_Transformer
from pytorch_memlab import profile
import datetime
import logging


# In[ ]:


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {'off', 'false', '0'}
    TRUTHY_STRINGS = {'on', 'true', '1'}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

# return True when only 1 item is True and others are all False
def bool_check_onlyone_stands(bools):
    length = len(bools)
    length_ls = list(range(length))
    for i in range(length):
        if bools[i]:
            copy_length_ls = copy.deepcopy(length_ls)
            copy_length_ls.pop(i)
            ex_is = copy_length_ls
            for e_i in ex_is:
                if bools[e_i]:
                    return False
            return True
        else:
            continue
    return False

def setup_logger(log_folder, modname=__name__):
    logger = getLogger(modname)
    logger.setLevel(DEBUG)

    sh = StreamHandler()
    sh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
#     logger.addHandler(sh)

    fh = FileHandler(log_folder) #fh = file handler
    fh.setLevel(DEBUG)
    fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger


# In[ ]:


parser = argparse.ArgumentParser()

# general setting
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--exp_id', type=str, default='1')
parser.add_argument('--trial_times', type=int, default=10)
parser.add_argument('--n_trial', type=int, default=1)
parser.add_argument('--tmp_path', type=str, default='/home/is/kosuke-t/log')
parser.add_argument('--dump_path', type=str, default='/ahc/work3/kosuke-t/SRHDA/XLM/log/')
parser.add_argument('--model_name', type=str, default='bert-base-uncased')
parser.add_argument('--langs', type=str, default='cs-en,de-en,fi-en,lv-en,ro-en,ru-en,tr-en,zh-en')
parser.add_argument('--empty_dump', type=bool_flag, default=False)
parser.add_argument('--train', type=bool_flag, default=True)
parser.add_argument('--test', type=bool_flag, default=True)

# hyperparameters
parser.add_argument('--batch_size', type=str, default='batch=8')
parser.add_argument('--epoch_size', type=int, default=3)
parser.add_argument('--optimizer', type=str, default='adam,lr=0.00001')
parser.add_argument('--lr_lambda', type=float, default=0.707, help='lr decay rate')
parser.add_argument('--dropout', type=float, default=0.0, help='mlp dropout')

# model setting
parser.add_argument('--amp', type=bool_flag, default=False)
parser.add_argument('--load_model', type=bool_flag, default=False)
parser.add_argument('--load_model_path', type=str, default='')
parser.add_argument('--save_model_name', type=str, default='model.pth')
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--hyp_ref', type=bool_flag, default=False)
parser.add_argument('--hyp_src', type=bool_flag, default=False)
parser.add_argument('--hyp_src_hyp_ref', type=bool_flag, default=False)
parser.add_argument('--hyp_src_ref', type=bool_flag, default=False)

# data setting
parser.add_argument('--model_path', type=str, default='', help='transformer model directory')
parser.add_argument('--src_train', type=str, default='', help='src data path for train')
parser.add_argument('--src_valid', type=str, default='', help='src data path for train')
parser.add_argument('--src_test', type=str, default='', help='src data path for train')
parser.add_argument('--ref_train', type=str, default='', help='ref data path for train')
parser.add_argument('--ref_valid', type=str, default='', help='ref data path for train')
parser.add_argument('--ref_test', type=str, default='', help='ref data path for train')
parser.add_argument('--hyp_train', type=str, default='', help='hyp data path for train')
parser.add_argument('--hyp_valid', type=str, default='', help='hyp data path for train')
parser.add_argument('--hyp_test', type=str, default='', help='hyp data path for train')
parser.add_argument('--label_train', type=str, default='', help='label data path for train')
parser.add_argument('--label_valid', type=str, default='', help='label data path for train')
parser.add_argument('--label_test', type=str, default='', help='label data path for train')
parser.add_argument('--darr', type=bool_flag, default=False, help='WMT17 is False otherwise True')

parser.add_argument('--train_shrink', type=float, default=1.0)
parser.add_argument('--debug', type=bool_flag, default=False)


args = parser.parse_args()

# make dump_path
args.dump_path = os.path.join(os.path.join(args.dump_path, args.exp_name), args.exp_id)
if not os.path.isdir(args.dump_path):
    os.makedirs(args.dump_path)
elif args.empty_dump:
    rmtree(args.dump_path)
    os.makedirs(args.dump_path)
if args.save_model_path == '':
    args.save_model_path = os.path.join(args.dump_path, args.save_model_name)
    
# make tmp_path
args.tmp_path = os.path.join(os.path.join(args.tmp_path, args.exp_name), args.exp_id)
if not os.path.isdir(args.tmp_path):
    os.makedirs(args.tmp_path)
elif args.empty_dump:
    rmtree(args.tmp_path)
    os.makedirs(args.tmp_path)
if args.save_model_path == '':
    args.save_model_path = os.path.join(args.tmp_path, args.save_model_name)

if not os.path.isdir(args.model_path):
    raise OSError
    print('model path :{} does not exist'.format(args.model_path))
args.model_path = os.path.join(args.model_path, args.model_name)
    
logging.basicConfig(filename=os.path.join(args.tmp_path, '{}.{}.log'.format(args.exp_name, datetime.date.today())), level=logging.DEBUG,  format="%(asctime)s %(levelname)-7s %(message)s")
args.logger = logging
# logger = setup_logger(os.path.join(args.tmp_path, '{}.{}.log'.format(args.exp_name, datetime.date.today())))
# args.logger = logger

args.langs = args.langs.split(',')

# bool_flags check
assert bool_check_onlyone_stands([args.hyp_ref, args.hyp_src, args.hyp_src_hyp_ref, args.hyp_src_ref])

# check if lang_ids is needed as inputs to the model
args.lang_id_bool = utils.check_for_langs_id(args.model_name)

src_train_flag = os.path.isfile(args.src_train) and os.path.isfile(args.src_valid)
src_test_flag = os.path.isfile(args.src_test)
ref_train_flag = os.path.isfile(args.ref_train) and os.path.isfile(args.ref_valid)
ref_test_flag = os.path.isfile(args.ref_test)
hyp_train_flag = os.path.isfile(args.hyp_train) and os.path.isfile(args.hyp_valid)
hyp_test_flag = os.path.isfile(args.hyp_test)
label_train_flag = os.path.isfile(args.label_train) and os.path.isfile(args.label_valid)
label_test_flag = os.path.isfile(args.label_test)
if args.train:
    if args.hyp_ref:
        assert hyp_train_flag and ref_train_flag
        args.forms = ['ref', 'hyp', 'label']
#         args.data_paths_train = [args.ref_train, args.hyp_train, args.label_train]
#         args.data_paths_valid = [args.ref_valid, args.hyp_valid, args.label_valid]
    elif args.hyp_src:
        assert hyp_train_flag and src_train_flag
        args.forms = ['src', 'hyp', 'label']
#         args.data_paths_train = [args.src_train, args.hyp_train, args.label_train]
#         args.data_paths_valid = [args.src_valid, args.hyp_valid, args.label_valid]
    elif args.hyp_src_hyp_ref or args.hyp_src_ref:
        assert hyp_train_flag and src_train_flag and ref_train_flag
        args.forms = ['src', 'ref', 'hyp', 'label']
    args.data_paths_train = [args.src_train, args.ref_train, args.hyp_train, args.label_train]
    args.data_paths_valid = [args.src_valid, args.ref_valid, args.hyp_valid, args.label_valid]
if args.test:
    if args.hyp_ref:
        assert hyp_test_flag and ref_test_flag
        args.forms = ['ref', 'hyp', 'label']
#         args.data_paths_test = [args.ref_test, args.hyp_test, args.label_test]
    elif args.hyp_src:
        assert hyp_test_flag and src_test_flag
        args.forms = ['src', 'hyp', 'label']
#         args.data_paths_test = [args.src_test, args.hyp_test, args.label_test]
    elif args.hyp_src_hyp_ref or args.hyp_src_ref:
        assert hyp_test_flag and src_test_flag and ref_test_flag
        args.forms = ['src', 'ref', 'hyp', 'label']
    args.data_paths_test = [args.src_test, args.ref_test, args.hyp_test, args.label_test]
        
    if not args.train:
        args.load_model = True
        if os.path.isfile(args.load_model_path):
            args.logger.info('ERRORL: model to be loaded does not exist!')
            exit(-2)
if not (args.train or args.test):
    args.logger.info('ERROR: argument -train or -test, either of th”’em must be true!')
    exit(-2)

if utils.get_model_type in ['bert']:
    args.use_token_type_ids = True
else:
    args.use_token_type_ids = False

args.batch_sizes = [int(x.split('=')[-1]) for x in args.batch_size.split('/')]
args.batch_size = args.batch_sizes[0]
args.optimizers = args.optimizer.split('/')

txt = ""
for key, value in args.__dict__.items():
    txt += '{}:{}{}'.format(str(key), str(value), os.linesep)
with open(os.path.join(args.tmp_path, 'arguments.txt'), mode='w', encoding='utf-8') as w:
    w.write(txt)


# In[ ]:


def update_lr(optimizer, args):
    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.lr_lambda
    return optimizer

# @profile
def run_model(model, batch, args, loss_fn, optimizer, train=False, getVector=False):
    if args.hyp_src_hyp_ref:
        x1 = batch['hyp_src'].to('cuda')
        if args.lang_id_bool:
            seg_x1 = batch['lang_hyp_src'].to('cuda')
        elif args.use_token_type_ids:
            seg_x1 = batch['seg_hyp_src'].to('cuda')
        x2 = batch['hyp_ref'].to('cuda')
        if args.lang_id_bool:
            seg_x2 = batch['lang_hyp_ref'].to('cuda')
        elif args.use_token_type_ids:
            seg_x2 = batch['seg_hyp_ref'].to('cuda')
        
        if args.lang_id_bool:
            h1 = model(x1, langs=seg_x1)[0][:,0,:]
            h2 = model(x2, langs=seg_x2)[0][:,0,:]
            
        elif args.use_token_type_ids:
            out1 = model(x1, token_type_ids=seg_x1)
            h1 = model(x1, token_type_ids=seg_x1)[1]
            h2 = model(x2, token_type_ids=seg_x2)[1]
        else:
            out1 = model(x1)
            h1 = model(x1)[1]
            h2 = model(x2)[1]
        h = torch.cat([h1,h2], dim=1)
    
    elif args.hyp_src:
        x1 = batch['hyp_src'].to('cuda')
        if args.lang_id_bool:
            seg_x1 = batch['lang_hyp_src'].to('cuda')
        elif args.use_token_type_ids:
            seg_x1 = batch['seg_hyp_src'].to('cuda')
        
        if args.lang_id_bool:
            h = model(x1, langs=seg_x1)[0]
        elif args.use_token_type_ids:
            h = model(x1, token_type_ids=seg_x1)[1]
        else:
            h = model(x1)[1]
        
    elif args.hyp_ref:
        x2 = batch['hyp_ref'].to('cuda')
        if args.lang_id_bool:
            seg_x2 = batch['lang_hyp_ref'].to('cuda')
        elif args.use_token_type_ids:
            seg_x2 = batch['seg_hyp_ref'].to('cuda')
        
        if args.lang_id_bool:
            h = model(x2, langs=seg_x2)[0]
        elif args.use_token_type_ids:
            h = model(x2, token_type_ids=seg_x2)[1]
        else:
            h = model(x2)[1]
            
    elif args.hyp_src_ref:
        x1 = batch['hyp_src_ref'].to('cuda')
        if args.lang_id_bool:
            seg_x1 = batch['lang_hyp_src_ref'].to('cuda')
        elif args.use_token_type_ids:
            seg_x1 = batch['seg_hyp_src_ref'].to('cuda')
        
        if args.lang_id_bool:
            h = model(x1, langs=seg_x1)[0]
        elif args.use_token_type_ids:
            h = model(x1, token_type_ids=seg_x1)[1]
        else:
            h = model(x1)[1]
    
    labels = batch['label'].to('cuda')
    preds = model.mlp(h)
    loss = loss_fn(preds.view(-1), labels.view(-1))
    
    if train:
        if not args.amp:
            loss.backward()
            optimizer.step()
        else:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
    preds = [float(p) for p in preds.view(-1).cpu().detach().numpy()]
    labels = [float(t) for t in labels.view(-1).cpu().detach().numpy()]
    hs = h.cpu().detach().numpy()
    
    if getVector:
        return float(loss.item()), preds, labels, hs
    else:
        return float(loss.item()), preds, labels


# In[ ]:


def _train(model, train_dataloader, mse, optimizer, args, results):
    model.train()
    losses = []
    preds_ls = []
    trues_ls = []
    raw_srcs = []
    raw_refs = []
    raw_hyps = []
    for n_iter, batch_data in enumerate(train_dataloader):
        if args.debug:
            args.logger.debug('\rnumber of iteration = {}'.format(n_iter), end='')
        optimizer.zero_grad()
        loss, preds, labels = run_model(model, batch_data, args, mse, optimizer, train=True)
        losses.append(loss)
        preds_ls.extend(preds)
        trues_ls.extend(labels)
        raw_srcs.extend(batch_data['raw_src'])
        raw_refs.extend(batch_data['raw_ref'])
        raw_hyps.extend(batch_data['raw_hyp'])

    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['loss'].append(np.mean(losses))
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'].append(utils.calc_pearson(preds_ls, trues_ls))
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pred'].append(preds_ls)
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['true'].append(trues_ls)
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_src'].append(raw_srcs)
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_ref'].append(raw_refs)
    results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_hyp'].append(raw_hyps)
    
    return model, train_dataloader, mse, optimizer, args, results


# In[ ]:


def _valid(model, valid_dataloader, mse, optimizer, args, results, 
           best_valid_loss, best_valid_pearson, n_epoch):
    model.eval()
    losses = []
    preds_ls = []
    trues_ls = []
    raw_srcs = []
    raw_refs = []
    raw_hyps = []
    for batch_data in valid_dataloader:
        with torch.no_grad():
            loss, preds, labels = run_model(model, batch_data, args, mse, optimizer)
        losses.append(loss)
        preds_ls.extend(preds)
        trues_ls.extend(labels)
        raw_srcs.extend(batch_data['raw_src'])
        raw_refs.extend(batch_data['raw_ref'])
        raw_hyps.extend(batch_data['raw_hyp'])

    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['loss'].append(np.mean(losses))
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'].append(utils.calc_pearson(preds_ls, trues_ls))
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pred'].append(preds_ls)
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['true'].append(trues_ls)
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_src'].append(raw_srcs)
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_ref'].append(raw_refs)
    results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['raw_hyp'].append(raw_hyps)

    # update lr
    if best_valid_loss > np.mean(losses):
        best_valid_loss = np.mean(losses)
    else:
        optimizer = update_lr(optimizer, args)

    #save model
    if best_valid_pearson['pearson'] < results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'][-1]:
        best_valid_pearson['pearson'] = results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'][-1]
        best_valid_pearson['optimizer'] = args.optimizer
        best_valid_pearson['batch_size'] = args.batch_size
        best_valid_pearson['epoch'] = n_epoch
        args.logger.info('saving a model!')
        checkpoint = {'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'amp': apex.amp.state_dict()}
        torch.save(checkpoint, os.path.join(args.tmp_path,'best_valid_checkpoint.pth'))
        args.logger.info('finished saving!')

    return model, valid_dataloader, mse, optimizer, args, results, best_valid_loss, best_valid_pearson


# In[ ]:


def _test(model, test_dataloader, mse, optimizer, args, results, lang_availables):
    model.eval()
    losses = {lang:[] for lang in args.langs}
    losses['all'] = []
    preds_ls = {lang:[] for lang in args.langs}
    preds_ls['all'] = []
    trues_ls = {lang:[] for lang in args.langs}
    trues_ls['all'] = []
    langs_ls = []
    raw_srcs = []
    raw_refs = []
    raw_hyps = []
    vectors = []
    for batch_data in test_dataloader:
        with torch.no_grad():
            loss, preds, labels, hs = run_model(model, batch_data, args, mse, optimizer, getVector=True)
        langs_ls.extend(batch_data['lang'])
        losses['all'].append(loss)
        preds_ls['all'].extend(preds)
        trues_ls['all'].extend(labels)
        raw_srcs.extend(batch_data['raw_src'])
        raw_refs.extend(batch_data['raw_ref'])
        raw_hyps.extend(batch_data['raw_hyp'])
        vectors.extend(hs)

    results['test']['loss'] = np.mean(losses['all'])
    results['test']['pred'] = preds_ls['all']
    results['test']['raw_src'] = langs_ls
    results['test']['raw_src'] = raw_srcs
    results['test']['raw_ref'] = raw_refs
    results['test']['raw_hyp'] = raw_hyps
    results['test']['vector'] = vectors
    
    if not args.darr:
        results['test']['pearson'] = utils.calc_pearson(preds_ls['all'], trues_ls['all'])
        results['test']['true'] = trues_ls['all']
        for lang, pred, true in zip(langs_ls, preds_ls['all'], trues_ls['all']):
            preds_ls[lang].append(pred)
            trues_ls[lang].append(true)
            if lang not in lang_availables:
                lang_availables.append(lang)
        for lang in lang_availables:
            results['test']['{}_pred'.format(lang)] = preds_ls[lang]
            results['test']['{}_true'.format(lang)] = trues_ls[lang]
            results['test']['{}_pearson'.format(lang)] = utils.calc_pearson(preds_ls[lang], trues_ls[lang])

    return model, test_dataloader, mse, optimizer, args, results, lang_availables


# In[1]:


# @profile
def _run_train(best_valid_pearson, 
               train_dataloader, valid_dataloader,
               args, results, 
               ModelClass, ConfigClass, 
               config):
    
    model = ModelClass.from_pretrained(args.model_path, config=config)
    model.config.num_labels = 1

    if args.hyp_src_hyp_ref:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size*2, 1)])
    else:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size, 1)])

    optimizer = utils.get_optimizer(list(model.parameters()), args.optimizer)
    mse = nn.MSELoss()
    model.to('cuda')

    if args.amp:
        model, optimizer = apex.amp.initialize(
            model,
            optimizer,
            opt_level=('O%i' % 1)
        )
    
    best_valid_loss = 1000
    
    for n_epoch in range(1, args.epoch_size+1):
        start_time = time.time()
        # train
        model, train_dataloader, mse, optimizer, args, results = _train(model, 
                                                                        train_dataloader, 
                                                                        mse, 
                                                                        optimizer, 
                                                                        args, 
                                                                        results)

        # valid
        model, valid_dataloader, mse, optimizer, args, results, best_valid_loss, best_valid_pearson = _valid(model, 
                                                                                                         valid_dataloader, 
                                                                                                         mse, optimizer, 
                                                                                                         args, 
                                                                                                         results, 
                                                                                                         best_valid_loss,
                                                                                                         best_valid_pearson,
                                                                                                             n_epoch)

        
        end_time = time.time()
        args.logger.info('exp_id:{}, n_trial:{}, epoch:{} finished!　　Took {}m{}s'.format(args.exp_id, args.n_trial, n_epoch, int((end_time-start_time)/60), int((end_time-start_time)%60)))
        args.logger.info('lr = {}'.format(optimizer.param_groups[0]['lr']))
        if args.train:
            args.logger.info('train loss_mean:{:.4f}, pearson:{:.4f}'.format(results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['loss'][-1],
                                                                             results['train'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'][-1]))
            args.logger.info('valid loss_mean:{:.4f}, pearson:{:.4f}'.format(results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['loss'][-1], 
                                                                             results['valid'][args.optimizer]['batch={}'.format(args.batch_size)][args.n_trial-1]['pearson'][-1]))
    
    with open(os.path.join(args.tmp_path, 'result.pkl'), mode='wb') as w:
        pickle.dump(results, w)
    
    return (best_valid_pearson, best_valid_epochs, results)


# In[ ]:


def _run_test(test_dataloader, ModelClass, config, results, args, lang_availables):
    
    checkpoint_path = ''
    if os.path.isfile(os.path.join(args.tmp_path,'best_valid_checkpoint.pth')):
        checkpoint_path = os.path.join(args.tmp_path,'best_valid_checkpoint.pth')
    elif os.path.isfile(os.path.join(args.dump_path,'best_valid_checkpoint.pth')):
        checkpoint_path = os.path.join(args.dump_path,'best_valid_checkpoint.pth')
    
    args.logger.info('loading the best model for testing!')
    checkpoint = torch.load(checkpoint_path)
    model = ModelClass.from_pretrained(args.model_path, config=config)

    if args.hyp_src_hyp_ref:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size*2, 1)])
    else:
        model.mlp = nn.Sequential(*[nn.Dropout(args.dropout),nn.Linear(model.config.hidden_size, 1)])

    model.to('cuda')
    optimizer = utils.get_optimizer(list(model.parameters()), args.optimizer)
    mse = nn.MSELoss()
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O%i' % 1)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    apex.amp.load_state_dict(checkpoint['amp'])

    args.logger.info('finished loading the model!')
    
    # test
    model, test_dataloader, mse, optimizer, args, results, lang_availables =  _test(model, 
                                                                                    test_dataloader, 
                                                                                    mse, 
                                                                                    optimizer, 
                                                                                    args, 
                                                                                    results,
                                                                                    lang_availables)  
        
    lang_availables = [l for l in args.langs if l in lang_availables]

    
    return results, lang_availables


# In[17]:


def main():
    TokenizerClass = utils.get_tokenizer_class(args.model_name)
    ModelClass = utils.get_model_class(args.model_name)
    ConfigClass = utils.get_config_class(args.model_name)
    tokenizer = TokenizerClass.from_pretrained(args.model_path)
    config = ConfigClass.from_pretrained(args.model_path)
    data_trans = Data_Transformer(args, tokenizer)
    args.model_config = config
    DATA = {}
    if args.train:
        DATA['train'] = Dataset(data_trans, tokenizer, args.data_paths_train, args, '{}.train'.format(args.exp_name))
        DATA['valid'] = Dataset(data_trans, tokenizer, args.data_paths_valid, args, '{}.valid'.format(args.exp_name))
        train_dataloader = torch.utils.data.DataLoader(DATA['train'], 
                                                   batch_size=args.batch_size, 
                                                   collate_fn=data_trans.collate_fn, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(DATA['valid'], batch_size=args.batch_size,
                                                   collate_fn=data_trans.collate_fn, shuffle=True)
    if args.test:
        DATA['test'] = Dataset(data_trans, tokenizer, args.data_paths_test, args, '{}.test'.format(args.exp_name))
    
        test_dataloader = torch.utils.data.DataLoader(DATA['test'], batch_size=args.batch_size,
                                                  collate_fn=data_trans.collate_fn, shuffle=False)
    
    result_path = os.path.join(args.tmp_path, 'result.pkl')
    if not os.path.isfile(result_path):
        results = {mode:
                       {opt:
                            {'batch={}'.format(bt):
                                 [{key:[] for key in ['loss', 'pearson', 'pred', 'true', 'raw_src', 'raw_ref', 'raw_hyp']} 
                                  for _ in range(args.trial_times)]
                             for bt in args.batch_sizes }
                        for opt in args.optimizers}
                   for mode in ['train', 'valid']}
        results['test'] = {key:[] for key in ['loss', 'pearson', 'pred', 'true', 'vector' 'raw_src', 'raw_ref', 'raw_hyp']}
    else:
        with open(result_path, mode='rb') as r:
            results = pickle.load(r) 

    best_valid_pearson_path = os.path.join(args.tmp_path, 'best_valid_pearson.pkl')
    if not os.path.isfile(best_valid_pearson_path):
        best_valid_pearson = {'optimizer':'', 'batch_size':0, 'n_trial':1, 'epoch':1, 'pearson':-1.0}
    else:
        with open(best_valid_pearson_path, mode='rb') as r:
            best_valid_pearson = pickle.load(r)
    
    lang_availables = [] # only for test
    
    if args.train:
        for opt in args.optimizers:
            args.optimizer = opt
            for bt in args.batch_sizes:
                args.batch_size = bt
                for n_trial in range(1, args.trial_times+1):
                    args.n_trial = n_trial
                    if len(results['valid'][opt]['batch={}'.format(bt)][args.n_trial-1]['pearson']) != args.epoch_size:
                        best_valid_pearson, results =  _run_train(best_valid_pearson, 
                                                                  train_dataloader, 
                                                                  valid_dataloader,
                                                                  args, results, ModelClass, ConfigClass, config)
                    with open(best_valid_pearson_path, mode='wb') as w:
                        pickle.dump(best_valid_pearson, w)
                         
        args.logger.info('Best Valid Pearson : {}'.format(best_valid_pearson['pearson']))            
        args.logger.info('Best Hyper-paramer : {}, batch={}, n_trial={}, epoch={}'.format(best_valid_pearson['optimizer'],
                                                                                          best_valid_pearson['batch_size'],
                                                                                          best_valid_pearson['n_trial'],
                                                                                          best_valid_pearson['epoch']))
    
    
    if args.test:
        args.logger.info('running test')
        resutls, lang_availables = _run_test(test_dataloader, ModelClass, config, results, args, lang_availables)
    with open(result_path, mode='wb') as w:
        pickle.dump(results, w) 
    args.logger.info('finished running test')
    if not args.darr:
        args.logger.info('--- Final Performance in Pearson---')
        txt = ""
        for lang in lang_availables:
            txt += '{} : {:.3f}'.format(lang, results['test']['{}_pearson'.format(lang)]) + str(os.linesep)
        txt += 'ave : {:.3f}'.format(np.mean([results['test']['{}_pearson'.format(lang)] for lang in lang_availables])) + str(os.linesep)
        txt += 'all : {:.3f}'.format(results['test']['pearson']) + str(os.linesep)
        args.logger.info(txt)
        performance_summary_filepath = os.path.join(args.tmp_path, 'final_pearformance.txt')
        with open(performance_summary_filepath, mode='w', encoding='utf-8') as w:
            w.write(txt)
    else:
        pass
    
    args.logger.info('moving tmp files to dump dir')
    for f in os.scandir(args.tmp_path):
        tmp_file = os.path.join(args.tmp_path, f.name)
        shutil.move(tmp_file, args.dump_path)


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:


# bash run.sh --exp_name multiBERT_all_hyp_ref --optimizer adam,lr=0.00003 --batch_size 16 --epoch_size 20 --hyp_src True --hyp_src_hyp_ref False

# bash run.sh --exp_name xlm-r-large_hyp_ref --hyp_ref True --model_name xlm-roberta-large --optimizer adam,lr=0.000007 adam,lr=0.000006 adam,lr=0.000005 adam,lr=0.000004 --epoch_size 10 --batch_size 2 --trial_times 10 --langs cs-en,de-en,lv-en,fi-en,ro-en,ru-en,tr-en,zh-en

