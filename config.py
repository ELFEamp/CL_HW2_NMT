# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/13 14:45 
'''

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
src_vocab_size = 32000
tgt_vocab_size = 32000
max_src_seq_length = 100
max_tgt_seq_length = 100
batch_size = 64
lr = 3e-4
share_enc_dec_weights = False
share_dec_proj_weights = True

data_dir = './data'
train_data_path = './data/train.json'
dev_data_path = './data/dev.json'
test_data_path = './data/test.json'