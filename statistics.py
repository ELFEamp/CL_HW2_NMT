# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：statistics.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/13 10:50 
'''

import json
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

if __name__ == "__main__":
	files = ['train', 'dev', 'test']
	en = english_tokenizer_load()
	ch = chinese_tokenizer_load()
	for file in files:
		with open('./data/'+file+'.json','r') as fr:
			corpus = json.load(fr)
			dic_en = {}
			dic_ch = {}
			for line in corpus:
				len_en = len(en.EncodeAsIds(line[0]))
				len_ch = len(ch.EncodeAsIds(line[1]))
				dic_en[len_en] = dic_en.get(len_en,0) + 1
				dic_ch[len_ch] = dic_ch.get(len_ch,0) + 1
		print(file,end=" ")
		print(dic_en,dic_ch)
