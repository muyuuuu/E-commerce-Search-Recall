import os
import torch.utils.data as data


class Supervised(data.Dataset):
    def __init__(self, data_tokenizer, root="./data/") -> None:
        super(Supervised, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, "corpus.tsv")
        self.train = os.path.join(self.root, "train.query.txt")
        self.corr = os.path.join(self.root, "qrels.train.tsv")
        self.split_char = "|||"
        self.all_data = {}
        self.train_data = {}
        self._create_train_data()
        self.tokenizer = data_tokenizer

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
        index = str(index + 1)
        anchor_text, pos_text = self.train_data[index].split(self.split_char)
        # tmp = random.randint(1, 1001492)
        # neg_text = self.all_data[str(tmp)]

        sample = self._post_process(anchor_text, pos_text)
        return sample

    def _post_process(self, anchor_text, pos_text):
        sample = self.tokenizer([anchor_text, pos_text],
                                truncation=True,
                                add_special_tokens=True,
                                max_length=120,
                                padding='max_length',
                                return_tensors='pt').to("cuda:0")

        return sample

    def __len__(self):
        return len(self.train_data)


class TESTDATA(data.Dataset):
    def __init__(self, data_tokenizer, root="./data/", certain="corpus.tsv") -> None:
        super(TESTDATA, self).__init__()
        self.root = root
        self.all = os.path.join(self.root, certain)
        self.all_data = {}
        self._create_eval_data()
        self.start = 0
        self.length = 120
        if certain != "corpus.tsv":
            self.start = 200000
        self.tokenizer = data_tokenizer

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
                              max_length=self.length,
                              padding='max_length',
                              return_tensors='pt').to("cuda:0")

        return id_, data

    def __len__(self):
        return len(self.all_data)
