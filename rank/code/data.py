import json
import os
import random
import collections
import tensorflow as tf
from tqdm import tqdm

from bert import tokenization


class InputExample(object):
    def __init__(self, qid, text_a, text_b=None, label=None):
        self.qid = qid  # qid##docid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def truncate_seq_pair(a_ids, b_ids, max_seq_length):
    while True:
        total_length = len(a_ids) + len(b_ids)
        if total_length <= max_seq_length:
            break
        if len(a_ids) > len(b_ids):
            a_ids.pop()
        else:
            b_ids.pop()


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    ids_a = example.text_a  # text_a is input_ids
    ids_b = example.text_b
    label_id = example.label
    qid_docid = example.qid
    truncate_seq_pair(ids_a, ids_b,
                      max_seq_length - 3)  # account for [CLS], [SEP], [SEP]

    tokens_a = tokenizer.convert_ids_to_tokens(ids_a)
    tokens_b = tokenizer.convert_ids_to_tokens(ids_b)
    tokens = []
    tokens.append("[CLS]")
    token_type_ids = []
    token_type_ids.append(0)
    for t in tokens_a:
        tokens.append(t)
        token_type_ids.append(0)
    tokens.append("[SEP]")
    token_type_ids.append(0)
    for t in tokens_b:
        tokens.append(t)
        token_type_ids.append(1)
    tokens.append("[SEP]")
    token_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(token_type_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        token_type_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("en_idx: %s" % (example.qid))
        tf.logging.info(
            "tokens: %s" %
            " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x)
                                                    for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            token_type_ids=token_type_ids,
                            label_id=label_id,
                            is_real_example=True)
    return feature


class DataProcessor(object):
    def __init__(self, corpus_ids_file, vocab_file):
        self.corpus_ids = self.read_ids_file(corpus_ids_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                                    do_lower_case=True)

    def read_ids_file(self, ids_file):
        count = 0
        ids_dict = dict()  # id(qid/ docid) -> ids
        print("*" * 10, "ids_file: ", ids_file)
        with open(ids_file) as f:
            for line in tqdm(f):
                line = json.loads(line)
                qid = line["qid"]
                input_ids = line["input_ids"]
                ids_dict[qid] = input_ids
                count += 1
        return ids_dict

    def get_train_examples(self, train_query_ids_file, trainset_dir):
        '''打标签的方式
        '''
        queryids_dict = self.read_ids_file(train_query_ids_file)
        files = os.listdir(trainset_dir)
        examples = []
        for file in files:
            if ".out" not in file:
                continue
            infile = os.path.join(trainset_dir, file)
            with open(infile) as f:
                for line in f:
                    # 查询 id 对应到语料库的 id
                    qid, golden_id, doc_ids = line.strip().split("\t")
                    if qid not in queryids_dict:
                        continue
                    # 拿到对应的 vocab_id
                    query_ids = queryids_dict[qid]
                    if golden_id not in self.corpus_ids:
                        continue
                    # 正样本
                    examples.append(
                        InputExample(qid + "##" + golden_id,
                                     text_a=query_ids,
                                     text_b=self.corpus_ids[golden_id],
                                     label=1))
                    # 负样本
                    chosed_docids = random.sample(doc_ids.split("#"), 2)
                    for docid in chosed_docids:
                        if docid not in self.corpus_ids:
                            continue
                        docids = self.corpus_ids[docid]
                        examples.append(
                            InputExample(qid + "##" + docid,
                                         text_a=query_ids,
                                         text_b=docids,
                                         label=0))
        return examples

    def get_features(self, examples, max_seq_length):
        features = []
        for (index, example) in enumerate(examples):
            feature = convert_single_example(index, example, max_seq_length,
                                             self.tokenizer)
            features.append(feature)
        return features

    def get_inputs(self, features):
        input_ids_lst = []
        input_mask_lst = []
        token_type_ids_lst = []
        label_ids_lst = []
        for feature in features:
            input_ids_lst.append(feature.input_ids)
            input_mask_lst.append(feature.input_mask)
            token_type_ids_lst.append(feature.token_type_ids)
            label_ids_lst.append(feature.label_id)
        return input_ids_lst, input_mask_lst, token_type_ids_lst, label_ids_lst
