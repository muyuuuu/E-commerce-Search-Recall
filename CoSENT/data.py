import os
import random
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
        self.all_data = {}
        self.train_data = {}
        self.data = []
        self._create_train_data()
        self.tokenizer = AutoTokenizer.from_pretrained(
            './pretrained/chinese-roberta-wwm-ext')

    def _create_train_data(self):
        # id -> train data sentence
        with open(self.train, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.train_data[line[0]] = line[1]
        # id -> all data sentence
        with open(self.all, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.all_data[line[0]] = line[1]
        # ('中州药膏', '中州氧化锌升华硫软膏20g痤疮酒渣鼻脂溢性皮炎', 1), 
        # ('灯芯绒裤子男秋季潮牌复古多口袋工装束脚裤纯棉微弹宽松休闲裤男', '汽车座套新老捷达新桑塔纳宝来老普桑朗逸polo亚麻全包座套布艺', 0)
        with open(self.corr, 'r') as f:
            length = len(self.all_data)
            for line in f.readlines():
                line = line.strip().split("\t")
                k = line[0]
                v = line[1]
                # pos data
                self.data.append((self.train_data[k], self.all_data[v], 1))
                # neg data
                neg1 = random.randint(1, length) % length
                neg2 = (neg1 + 1) % length
                if neg1 == 0:
                    neg1 = 1
                    neg2 = 2
                self.data.append(
                    (self.all_data[str(neg1)], self.all_data[str(neg2)], 0))

    def __getitem__(self, index):
        sen1, sen2, label = self.data[index]
        sen1, sen2 = self._post_process(sen1, sen2)
        return sen1, sen2, torch.tensor(label).long()

    def _post_process(self, sen1, sen2):
        sen1 = self.tokenizer(sen1,
                              truncation=True,
                              add_special_tokens=True,
                              max_length=48,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")
        sen2 = self.tokenizer(sen2,
                              truncation=True,
                              add_special_tokens=True,
                              max_length=48,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return sen1, sen2

    def __len__(self):
        return len(self.data)


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
        id_, text = str(index + self.start +
                        1), self.all_data[str(index + self.start + 1)]
        data = self.tokenizer(text,
                              truncation=True,
                              add_special_tokens=True,
                              max_length=48,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return id_, data

    def __len__(self):
        return len(self.all_data)


if __name__ == "__main__":
    a = TEXTDATA()
