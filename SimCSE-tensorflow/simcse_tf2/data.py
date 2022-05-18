# -*- coding:utf-8 -*-
"""
Author:
    jiei, jifei@outlook.com
"""
import os

os.environ["TF_KERAS"] = '1'
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import DataGenerator, sequence_padding
import random
import numpy as np


def get_tokenizer(dict_path):
    """build tokenizer
    """
    return Tokenizer(dict_path, do_lower_case=True)


def texts_to_ids(texts, dict_path, max_len=64):
    """texts to ids
    """
    tokenizer = get_tokenizer(dict_path)
    token_ids = []
    for t in texts:
        token_ids.append(tokenizer.encode(t, maxlen=max_len)[0])
    return sequence_padding(token_ids)


class SimCseDataGenerator(DataGenerator):
    """Data Generator

    """

    def __init__(self, data, dict_path, batch_size=32, max_len=64, text_tuple_size=2, buffer_size=None):

        super().__init__(data, batch_size, buffer_size)
        assert text_tuple_size in [1, 2, 3]
        self.tokenizer = get_tokenizer(dict_path)
        self.max_len = max_len
        self.text_tuple_size = text_tuple_size

    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, texts in self.sample(True):
            if self.text_tuple_size == 1:  # unsupervised one text,repeat self
                batch_token_ids.append(self.tokenizer.encode(texts[0], maxlen=self.max_len)[0])
                batch_token_ids.append(self.tokenizer.encode(texts[0], maxlen=self.max_len)[0])
            elif self.text_tuple_size == 2:  # texts pair
                batch_token_ids.append(self.tokenizer.encode(texts[0], maxlen=self.max_len)[0])
                batch_token_ids.append(self.tokenizer.encode(texts[1], maxlen=self.max_len)[0])
            else:  # negative sampling
                batch_token_ids.append(self.tokenizer.encode(texts[0], maxlen=self.max_len)[0])
                batch_token_ids.append(self.tokenizer.encode(texts[1], maxlen=self.max_len)[0])
                batch_token_ids.append(self.tokenizer.encode(texts[2], maxlen=self.max_len)[0])

            if len(batch_token_ids) == self.batch_size * self.text_tuple_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def load_data(file_name, delimiter='\t', skip_header=False, shuffle=True, random_negative_sampling=False):
    """ Load data from file

    :param file_name:string, file path
    :param skip_header:bool, need skip first line
    :param delimiter:string
    :param shuffle:bool, shuffle data
    :param random_negative_sampling: bool, Random Negative Sampling.
    :return:list,[(text1),...] or [(text1, text2),...] or [(text1, text2, neg text),...]
    """
    lines = []
    negs = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            if skip_header:
                skip_header = False
            else:
                columns = line.strip().split(delimiter)
                lines.append(tuple(columns))
                if random_negative_sampling and len(columns) == 2:
                    negs.append(columns[1])

    if shuffle:
        random.shuffle(lines)
    if random_negative_sampling:
        random.shuffle(negs)
        return [(i[0], i[1], negs.pop()) for i in lines]
    return lines
