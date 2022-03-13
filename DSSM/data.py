import os

from numpy import indices
import torch
import torch.utils.data as data
from transformers import AutoTokenizer


class TEXTDATA(data.Dataset):
    def __init__(self, root="./data/") -> None:
        super(TEXTDATA, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.train = os.path.join(self.root, "train.query.txt")
        self.corr = os.path.join(self.root, "qrels.train.tsv")
        self.split_char = "|||"
        self.all_data = {}
        self.train_data = {}
        self._create_train_data()
        self.neg_num = 10
        self.tokenizer = AutoTokenizer.from_pretrained(
            './pretrained/chinese-roberta-wwm-ext')

    def _create_train_data(self):
        with open(self.train, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.train_data[line[0]] = line[1] + self.split_char
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]
        with open(self.corr, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                k = line[0]
                v = line[1]
                self.train_data[k] += self.all_data[v]

    def __getitem__(self, index):
        tmp = index + 1
        index = str(index + 1)
        anchor_text, pos_text = self.train_data[index].split(self.split_char)
        neg_text = []
        for i in range(tmp, tmp + self.neg_num):
            neg_text.append(self.all_data[str(i)])

        assert len(neg_text) == 10
        anchor, pos, neg = self._post_process(anchor_text, pos_text, neg_text)
        return anchor, pos, neg

    def _post_process(self, anchor_text, pos_text, neg_text):
        anchor = self.tokenizer(anchor_text,
                                add_special_tokens=True,
                                max_length=128,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")
        pos = self.tokenizer(pos_text,
                             add_special_tokens=True,
                             max_length=128,
                             padding='max_length',
                             return_tensors='pt').to("cuda:0")

        neg = []
        for i in range(len(neg_text)):
            content_encoding = self.tokenizer([neg_text[i]],
                                              add_special_tokens=True,
                                              max_length=128,
                                              padding='max_length',
                                              return_tensors='pt').to("cuda:0")
            neg.append(content_encoding)

        return anchor, pos, neg

    def __len__(self):
        return len(self.train_data)


class TESTDATA(data.Dataset):
    def __init__(self, root="./data/", certain="corpus.tsv") -> None:
        super(TESTDATA, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, certain)
        self.all_data = {}
        self._create_eval_data()
        self.start = 0
        if certain != "corpus.tsv":
            self.start = 200000
        self.tokenizer = AutoTokenizer.from_pretrained(
            './pretrained/chinese-roberta-wwm-ext')

    def _create_eval_data(self):
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]

    def __getitem__(self, index):
        id_, text = str(index + self.start + 1), self.all_data[str(index + self.start + 1)]
        data = self.tokenizer(text,
                              add_special_tokens=True,
                              max_length=128,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return id_, data

    def __len__(self):
        return len(self.all_data)


if __name__ == "__main__":
    a = TEXTDATA()
