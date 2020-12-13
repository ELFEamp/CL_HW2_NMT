# CL_HW2_NMT
Homework2 of Computational Linguistics -- NMT(en-ch)

### 数据处理

1. 分词

   * 工具：sentencepiece包

   * 预处理：`./data/get_corpus.py`抽取train、dev和test中双语语料，分别保存到`corpus.en`和`corpus.ch`中，每行一个句子

   * 训练分词模型：`./tokenizer/tokenize.py`中调用了sentencepiece.SentencePieceTrainer.Train()方法，利用`corpus.en`和`corpus.ch`中的语料训练分词模型，训练完成后会在`./tokenizer`文件夹下生成`chn.model`,`chn.vocab`，`eng.model`和`eng.vocab`，其中`.model`和`.vocab`分别为模型文件和对应的词表

   * 分词模型的使用（以中文为例）：

     ```python
     import sentencepiece as spm
     sp_chn = spm.SentencePieceProcessor()
     sp_chn.Load('{}.model'.format("./tokenizer/chn"))
     sentence = "美国总统特朗普"
     print(sp_chn.EncodeAsIds(sentence))
     ```

     输出

     ```
     [12907, 277]
     ```

### TODO

1. 模型的transformer部分直接使用了[如下repo的实现](https://github.com/jungwhank/transformer-pl)，但存在一些问题。比如：
   * 该模型（好像）限制了输入输出的sequence（分词后的）长度保持一致，不然会报错；
   * 所有batch的sequece都必须pad到max_seq_length而不是batch里的最大值
2. evaluate部分，BLUE