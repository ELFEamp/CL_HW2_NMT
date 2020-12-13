# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/10 22:04 
'''
import torch
import utils
from argparse import ArgumentParser
from data_loader import MTDataset
from torch.utils.data import DataLoader
from model import Transformer
from train import train

def get_config():
	parser = ArgumentParser()
	parser.add_argument('--d_model', type=int, default=512)
	parser.add_argument('--n_heads', type=int, default=8)
	parser.add_argument('--n_layers', type=int, default=6)
	parser.add_argument('--d_k', type=int, default=64)
	parser.add_argument('--d_v', type=int, default=64)
	parser.add_argument('--d_ff', type=int, default=2048)
	parser.add_argument('--dropout', type=float, default=0.1)

	parser.add_argument('--padding_idx', type=int, default=0)
	parser.add_argument('--src_vocab_size', type=int, default=32000)
	parser.add_argument('--tgt_vocab_size', type=int, default=32000)
	parser.add_argument('--max_src_seq_length', type=int, default=50)
	parser.add_argument('--max_tgt_seq_length', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--epoch_num', type=int, default=10)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--share_enc_dec_weights', type=bool, default=False)
	parser.add_argument('--share_dec_proj_weights', type=bool, default=True)

	parser.add_argument('--data_dir', type=str, default='./data')
	parser.add_argument('--train_data_path', type=str, default='./data/train.json')
	parser.add_argument('--dev_data_path', type=str, default='./data/dev.json')
	parser.add_argument('--test_data_path', type=str, default='./data/test.json')
	parser.add_argument('--log_path', type=str, default='./experiment/train.log')

	parser.add_argument('--gpu', type=str, default='')

	return vars(parser.parse_args())

if __name__ == "__main__":
	config = get_config()

	# set device
	if config.get('gpu', '') != '':
		device = torch.device(f"cuda:{config['gpu']}")
	else:
		device = torch.device('cpu')

	utils.set_logger(config["log_path"])

	train_dataset = MTDataset(config['train_data_path'], config['max_src_seq_length'], config['max_tgt_seq_length'])
	dev_dataset = MTDataset(config['dev_data_path'], config['max_src_seq_length'], config['max_tgt_seq_length'])
	test_dataset = MTDataset(config['test_data_path'], config['max_src_seq_length'], config['max_tgt_seq_length'])

	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'],
	                              collate_fn=train_dataset.collate_fn)
	dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config['batch_size'],
	                            collate_fn=dev_dataset.collate_fn)
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'],
	                             collate_fn=test_dataset.collate_fn)

	model = Transformer(config)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
	train(model, optimizer, device, config, train_dataloader, dev_dataloader)
	# train()
