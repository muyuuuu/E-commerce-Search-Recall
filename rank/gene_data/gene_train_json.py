import os

os.environ["TF_KERAS"] = '1'
from bert4keras.snippets import sequence_padding
from simcse_tf2.simcse import simcse
from simcse_tf2.data import get_tokenizer, load_data, SimCseDataGenerator
from simcse_tf2.losses import simcse_loss
import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
import json


def texts_to_ids(data, tokenizer, max_len=64):
    """转换文本数据为id形式
    """
    token_ids = []
    for d in data:
        token_ids.append(tokenizer.encode(d, maxlen=max_len)[0])
    return sequence_padding(token_ids)


model_path = '/home/20031211375/tf-simcse/ecom'
checkpoint_path = '%s/bert_model.ckpt' % model_path
config_path = '%s/bert_config.json' % model_path
dict_path = '%s/vocab.txt' % model_path
train_query = [line[0] for line in csv.reader(open("./data/train.txt"), delimiter='\t')]
train_idx = [str(line) for line in range(1, 100001)]
pre_batch_size = 5000
tokenizer = get_tokenizer(dict_path)

data = []
for i in tqdm(range(0, len(train_query), pre_batch_size)):
    batch_text = train_query[i:i + pre_batch_size]
    batch_ids = train_idx[i:i + pre_batch_size]
    print("query size:", len(batch_text))
    temp_embedding = texts_to_ids(batch_text, tokenizer, max_len=150)
    for j in range(len(temp_embedding)):
        writer_str = temp_embedding[j].tolist()
        writer_str = [i for i in writer_str if i != 0][1:-1]
        tmp = {}
        tmp["qid"] = batch_ids[j]
        tmp["input_ids"] = writer_str
        data.append(tmp)

with open("train.query.json", 'w') as f:
    for i in data:
        json.dump(i, f)
        f.write("\n")

