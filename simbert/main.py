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
import faiss
# from faiss import  IndexFlatIP
from sklearn.preprocessing import normalize
import logging
import sys

from keras.utils import multi_gpu_model

logging.basicConfig(level=logging.INFO)
jieba.initialize()

# 基本信息 需要改动
epochs = 50
save_dir = "./output/res_finetune_unsupervised_100w_mcpr_random_rate/"
model_dir = "./1-chinese_roformer-sim-char_L-12_H-768_A-12/chinese_roformer-sim-char_L-12_H-768_A-12/"
model_type = "roformer"
# 不需要改动的变量
maxlen = 70
batch_size =  80 # 调整
num_dim = 128
seq2seq_loss_ratio = 0.5
train_data_path = "./model_data/hold_out_t2s/train.txt" # 训练集 que-doc
dev_data_path = "./model_data/hold_out_t2s/dev.txt" # 验证集que-doc
# corpus_path = "./model_data/hold_out_t2s/dev.txt"

corpus_path = "model_data/hold_out_t2s/mini_corpus_clean.txt"
full_corpus_path = "model_data/hold_out_t2s/corpus_clean.txt"
unsupervised_data_path = "/home/zqxie/project/SimCSE-Chinese-Pytorch-main/data/unsupervised_data.txt"
mcpr_path = '/home/zqxie/project/SimCSE-Chinese-Pytorch-main/data/mcpr_train.txt'

# bert配置
config_path = join(model_dir, 'bert_config.json')
checkpoint_path = join(model_dir, 'bert_model.ckpt')
dict_path = join(model_dir, 'vocab.txt')
steps_per_epoch = 1000000 // batch_size
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def corpus():
    """读取语料
    """
    while True:
        data = []
        with open(train_data_path, "r", encoding="utf8") as fr: # 为什么他这里没有考虑que-doc的对应关系？
            t = [line.strip().split("\t") for line in fr]
            for _ in range(1): data.extend(t)
        with open(dev_data_path, "r", encoding="utf8") as fr:
            t = [line.strip().split("\t") for line in fr]
            for _ in range(1): data.extend(t)
        with open(unsupervised_data_path, "r", encoding="utf8") as fr:
            t = [line.strip().split("\t") for line in fr]
            for _ in range(1): data.extend(t)
        with open(mcpr_path, "r", encoding="utf8") as fr:
            t = [line.strip().split("\t") for line in fr]
            for _ in range(1): data.extend(t)

        print("总量数据：{}".format(len(data)))
        random.shuffle(data)
        for item in data:
            yield item


def masked_encode(text): ## 作用是什么
    """wwm随机mask
    """
    words = jieba.lcut(text)
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0] #
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:maxlen - 1] + [tokenizer._token_end_id]
    target = target[:maxlen - 1] + [0]
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (text, synonym) in self.sample(random):
            for i in range(2):
                if np.random.random() < 0.5:
                    text_ids = masked_encode(text)[0] # 只针对que文档, 返回的source部分
                else:
                    text_ids = tokenizer.encode(text)[0] # 用tokenizer解码和mask解码的区别
                synonym_ids = tokenizer.encode(synonym)[0][1:]
                truncate_sequences(maxlen * 2, -2, text_ids, synonym_ids)
                token_ids = text_ids + synonym_ids
                segment_ids = [0] * len(text_ids) + [1] * len(synonym_ids)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                self.some_samples.append(synonym)
                if len(self.some_samples) > 1000:
                    self.some_samples.pop(0)
                text, synonym = synonym, text
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class TotalLoss(Loss): # 看一下sim_bert_v2 重写的loss的哪一部分
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """

    def compute_loss(self, inputs, mask=None): # 这里的mask怎么传入的？
        # 确认一下input的具体内容
        loss1 = self.compute_loss_of_seq2seq(inputs, mask) * seq2seq_loss_ratio # loss的 权重
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = K.l2_normalize(y_pred, axis=1)  # 句向量归一化
        similarities = K.dot(y_pred, K.transpose(y_pred))  # 相似度矩阵
        similarities = similarities - K.eye(K.shape(y_pred)[0]) * 1e12  # 排除对角线
        similarities = similarities * 20  # scale
        loss = K.categorical_crossentropy(
            y_true, similarities, from_logits=True
        )
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels


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

outputs = TotalLoss([2, 3])(roformer.inputs + roformer.outputs)
### 确认roformer的input 和output
# print('inputs:{} outputs:{}'.format(roformer.inputs, roformer.outputs))
model = keras.models.Model(roformer.inputs, outputs) # 这里的output的用法

model = multi_gpu_model(model, gpus=2)

AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=1e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.summary()


class Evaluate(keras.callbacks.Callback):
    """评估模型
    """

    def __init__(self):
        self.best_mrr = -1
        self.mrrs = []
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        roformer.save_weights_as_checkpoint(join(save_dir, "best_model/bert_model.ckpt"))
        copy(config_path, join(save_dir, "best_model/bert_config.json"))
        copy(dict_path, join(save_dir, "best_model/vocab.txt"))
        # if epoch > 25:
        #     # 保存最优
        #     mrr, acc = self.evaluate(epoch)
        #     self.mrrs.append(mrr)
        #     logging.info('mrr:{:.4f} acc:{:.4f} best_mrr:{:.4f} best_epoch:{}'.format(mrr, acc, self.best_mrr, self.best_epoch))
        #     if self.best_mrr <= mrr:
        #         self.best_mrr = mrr
        #         self.best_epoch = epoch
        #         roformer.save_weights_as_checkpoint(join(save_dir, "best_model/bert_model.ckpt"))
        #         copy(config_path, join(save_dir, "best_model/bert_config.json"))
        #         copy(dict_path, join(save_dir, "best_model/vocab.txt"))
        #     # 早停
        #     if len(self.mrrs) > 1 and self.mrrs[-1] < self.mrrs[-2] and acc > 0.8:
        #         logging.info("train的mrr下降，终止训练")
        #         sys.exit()

    def evaluate(self, epoch):
        logging.info("开始评测")
        with open(dev_data_path, "r", encoding="utf8") as fr:
            test_data = [line.strip().split("\t") for line in fr]  # query, title
        with open(corpus_path, "r", encoding="utf8") as fr:
            corpus_titles = [line.strip().split("\t")[-1] for line in fr]  # title
        # 获取向量
        query_vecs = self.get_vecs([i[0] for i in test_data])
        corpus_titles_vecs = self.get_vecs(corpus_titles)

        logging.info('index get over...')
        # 构建faiss索引
        faiss_index = faiss.IndexFlatIP(corpus_titles_vecs.shape[1])
        faiss_index.add(corpus_titles_vecs)
        # 搜索结果
        res_distance, res_index = faiss_index.search(query_vecs, 10)
        # 计算指标
        top10acc, mrr10 = 0.0, 0.0
        for i in range(res_index.shape[0]):
            query, title = test_data[i]
            topk_titles = [corpus_titles[res_index[i, j]] for j in range(res_index.shape[1])]
            if title in topk_titles:
                top10acc += 1
                mrr10 += (1.0 / (1 + topk_titles.index(title)))
        res = {"top10acc": top10acc / len(test_data), "mrr@10": mrr10 / len(test_data), "epoch": epoch}
        logging.info(str(res))
        return mrr10 / len(test_data), top10acc / len(test_data)

    def get_vecs(self, sens):
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
                x, s = tokenizer.encode(t, maxlen=maxlen)
                X.append(x)
                S.append(s)
            start += batch_size_tmp
            X = sequence_padding(X)
            S = sequence_padding(S)
            vecs.append(encoder.predict([X, S]))
        return normalize(np.vstack(vecs)[:, :num_dim], norm="l2", axis=1)


if __name__ == '__main__':
    train_generator = data_generator(corpus(), batch_size)
    evaluator = Evaluate()
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )
