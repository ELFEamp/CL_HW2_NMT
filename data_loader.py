# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：data_loader.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/13 1:00 
'''
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

class MTDataset(Dataset):
	def __init__(self, data_path, max_src_seq_length, max_tgt_seq_length):
		self.dataset = json.load(open(data_path, 'r'))
		self.sp_eng = english_tokenizer_load()
		self.sp_chn = chinese_tokenizer_load()
		self.PAD = self.sp_chn.pad_id()
		self.UNK = self.sp_chn.unk_id()
		self.BOS = self.sp_chn.bos_id()
		self.EOS = self.sp_chn.eos_id()
		self.max_src_seq_length = max_src_seq_length
		self.max_tgt_seq_length = max_tgt_seq_length

	def __getitem__(self, idx):
		eng_text = self.dataset[idx][0]
		chn_text = self.dataset[idx][1]
		return [eng_text, chn_text]

	def __len__(self):
		return len(self.dataset)

	def getTensor(self, src_tokens, tgt_tokens):
		batch_size = len(src_tokens)
		batch_input = torch.LongTensor(batch_size, self.max_src_seq_length).fill_(0)
		batch_target = torch.LongTensor(batch_size, self.max_tgt_seq_length+1).fill_(0)

		for i in range(batch_size):
			batch_input[i,:len(src_tokens[i])] = torch.LongTensor(src_tokens[i])
			batch_target[i,:len(tgt_tokens[i])] = torch.LongTensor(tgt_tokens[i])

		return batch_input, batch_target


	def collate_fn(self, batch):

		src_text = [x[0] for x in batch]
		tgt_text = [x[1] for x in batch]

		src_tokens = [self.sp_eng.EncodeAsIds(sent)[:self.max_src_seq_length] for sent in src_text]
		tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent)[:self.max_tgt_seq_length-1] + [self.EOS] for sent in tgt_text]

		batch_input, batch_target = self.getTensor(src_tokens, tgt_tokens)
		# batch_input = pad_sequence([torch.LongTensor(np.array(tids)) for tids in src_tokens],
		#                            batch_first=True, padding_value=self.PAD)
		# batch_target = pad_sequence([torch.LongTensor(np.array([self.BOS]+tids+[self.EOS])) for tids in tgt_tokens],
		#                             batch_first=True, padding_value=self.PAD)

		return {"input":batch_input, "target":batch_target}
if __name__ == "__main__":
	dataset = MTDataset('./data/test.json',100,100)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)
	# for batch in dataloader:
	# 	print(batch)