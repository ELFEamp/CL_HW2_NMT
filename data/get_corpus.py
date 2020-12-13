# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：get_corpus.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/12 22:57 
'''
import json

if __name__ == "__main__":
	files = ['train', 'dev', 'test']
	ch_path = 'corpus.ch'
	en_path = 'corpus.en'
	ch_lines = []
	en_lines = []

	for file in files:
		corpus = json.load(open(file+'.json','r'))
		for item in corpus:
			ch_lines.append(item[1]+'\n')
			en_lines.append(item[0]+'\n')

	with open(ch_path, "w") as fch:
		fch.writelines(ch_lines)

	with open(en_path, "w") as fen:
		fen.writelines(en_lines)