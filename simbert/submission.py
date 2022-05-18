#! -*- coding: utf-8 -*-
# SimBERT v2 训练代码
# 只使用train-dev
import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import random
from os.path import join
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import text_segmentate, truncate_sequences
import jieba
from shutil import copy
# import faiss
from faiss import  IndexFlatIP
from sklearn.preprocessing import normalize
import logging
import sys
import csv
from keras.utils import multi_gpu_model
from tqdm import  tqdm

logging.basicConfig(level=logging.INFO)
jieba.initialize()

# 基本信息 需要改动
# epochs = 25
# save_dir = "./output/res_finetune_mcpr"
model_dir = "./output/res_finetune_mcpr_L_after_25_epoch/best_model/"
model_type = "roformer"
# 不需要改动的变量
maxlen = 70
batch_size =  80 # 调整
num_dim = 128
seq2seq_loss_ratio = 0.5
train_data_path = "./model_data/hold_out_t2s/train.txt" # 训练集 que-doc
dev_data_path = "./model_data/hold_out_t2s/dev.txt" # 验证集que-doc
# corpus_path = "./model_data/hold_out_t2s/dev.txt"
dev_data_clean_path = './model_data/hold_out_t2s/dev.query_clean.txt'

corpus_path = "model_data/hold_out_t2s/mini_corpus_clean.txt"
full_corpus_path = "model_data/hold_out_t2s/corpus_clean.txt"
# mcpr_label_data_path = "./model_data/mcpr/processed/labeled_data.txt"
# bert配置
config_path = join(model_dir, 'bert_config.json')
checkpoint_path = join(model_dir, 'bert_model.ckpt')
dict_path = join(model_dir, 'vocab.txt')
steps_per_epoch = 200000 // batch_size
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model=model_type,
    application='unilm',
    with_pool='linear',
    with_mlm='linear',
    dropout_rate=0.2,
    ignore_invalid_weights=True,
    return_keras_model=False,
)

encoder = keras.models.Model(roformer.inputs, keras.layers.Lambda(lambda x: x[:, :num_dim])(roformer.outputs[0]))
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])


def evaluate():
    logging.info("开始评测")
    with open(dev_data_clean_path, "r", encoding="utf8") as fr:
        test_data = [line.strip().split("\t")[-1] for line in fr]  # query, title
    with open(full_corpus_path, "r", encoding="utf8") as fr:
        corpus_titles = [line.strip().split("\t")[-1] for line in fr]  # title
    # 获取向量
    a_vecs = get_vecs(test_data)
    query_embedding_file = csv.writer(open('./query_embedding', 'w'), delimiter='\t')
    batch_size = 1

    print('a_vec[0] len:{}'.format(a_vecs.shape))
    for i in tqdm(range(0, len(a_vecs), batch_size)):
        writer_str = a_vecs[i].tolist()
        writer_str = [format(s, '.8f') for s in writer_str]
        writer_str = ','.join(writer_str)
        query_embedding_file.writerow([i + 200001, writer_str])

    print('wirte queries over...')

    doc_embedding_file = csv.writer(open('./doc_embedding', 'w'), delimiter='\t')
    b_vecs = get_vecs(corpus_titles)
    print('begin to write doc embedding...')
    for i in tqdm(range(0, len(b_vecs), batch_size)):
        writer_str = b_vecs[i].tolist()
        writer_str = [format(s, '.8f') for s in writer_str]
        writer_str = ','.join(writer_str)
        doc_embedding_file.writerow([i + 1, writer_str])
    # print('query shape:{}'.format(b_vecs[0]))

def get_vecs(sens):
    """
    获取归一化后的句向量
    :param sens:
    :return:
    """
    vecs, start, = [], 0
    batch_size_tmp = batch_size // 2
    while start < len(sens):
        # print("获取句向量的进度：{}".format(start / len(sens)))
        X, S = [], []
        for t in sens[start:start + batch_size_tmp]:
            # print('t is :{}'.format(t))
            x, s = tokenizer.encode(t, maxlen=maxlen) # 怎么会返回两个结果呢==》 (token_id, segment_id)
            X.append(x)
            S.append(s)
        start += batch_size_tmp
        X = sequence_padding(X)
        S = sequence_padding(S)
        vecs.append(encoder.predict([X, S]))
    return normalize(np.vstack(vecs)[:, :num_dim], norm="l2", axis=1)

if __name__ == '__main__':
    evaluate()
