# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/13 14:45 
'''

import torch.nn.functional as F
import torch.nn as nn
import logging
from tqdm import tqdm

def train(model, optimizer, device, config, train_loader, dev_loader):
	model.to(device)
	loss_fnt = nn.CrossEntropyLoss(ignore_index=config['padding_idx'])
	for epoch in range(config['epoch_num']):
		model.train()
		for idx, batch in tqdm(enumerate(train_loader)):
			src = batch['input']
			tgt = batch['target']

			src = src.to(device)
			tgt = tgt.to(device)

			tgt_label = tgt[:, 1:]
			tgt_hat, *_ = model.forward(src, tgt[:, :-1])
			loss = loss_fnt(tgt_hat, tgt_label.contiguous().view(-1))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		# print(loss)
		logging.info("Epoch: {},dev loss: {}".format(epoch, ))

def evaluate(dev_loader, model, mode='dev'):
