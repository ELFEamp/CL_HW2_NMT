# -*- coding: UTF-8 -*-
'''
@Project ：cl_hw2 
@File    ：tokenize.py
@IDE     ：PyCharm 
@Author  ：elfe
@Date    ：2020/12/12 23:13 
'''

import sentencepiece as spm

def train(input_file, vocab_size, model_name, model_type, character_coverage):
	'''
	search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
	--input: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.
	--model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
	--vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
	--character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanse or Chinese and 1.0 for other languages with small character set.
	--model_type: model type. Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.
	:param input_file:
	:param vocab_size:
	:param model_name:
	:param model_type:
	:param character_coverage:
	:return:
	'''
	input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
	cmd = input_argument%(input_file, model_name, vocab_size, model_type, character_coverage)
	spm.SentencePieceTrainer.Train(cmd)

if __name__ == "__main__":
	en_input = '../data/corpus.en'
	en_vocab_size = 32000
	en_model_name = 'eng'
	en_model_type = 'bpe'
	en_character_coverage = 1
	train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

	ch_input = '../data/corpus.ch'
	ch_vocab_size = 32000
	ch_model_name = 'chn'
	ch_model_type = 'bpe'
	ch_character_coverage = 0.9995
	train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)