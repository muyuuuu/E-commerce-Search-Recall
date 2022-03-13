import torch
from transformers import AutoModel
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='./pretrained/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        self.fc = torch.nn.Linear(768, output_dim)
        self.dropout = torch.nn.Dropout(p=0.1)

    def extract(self, sen):
        # max pool to get feature
        attention_mask = sen['attention_mask']
        last_hidden_state = self.dropout(self.extractor(**sen)[0])
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        logits = self.fc(max_embeddings)
        logits = F.normalize(logits, p=2, dim=-1)
        return logits

    def forward(self, sen1, sen2, labels):
        sen1["input_ids"] = sen1["input_ids"].squeeze()
        sen1["attention_mask"] = sen1["attention_mask"].squeeze()
        sen1["token_type_ids"] = sen1["token_type_ids"].squeeze()
        sen2["input_ids"] = sen2["input_ids"].squeeze()
        sen2["attention_mask"] = sen2["attention_mask"].squeeze()
        sen2["token_type_ids"] = sen2["token_type_ids"].squeeze()

        sen1, sen2 = self.extract(sen1), self.extract(sen2)
        dis_sim = torch.sqrt(torch.sum((sen1 - sen2).pow(2), axis=-1)) * 20
        dis_sim = dis_sim[:, None] - dis_sim[None, :]

        labels = labels[:, None] > labels[None, :]
        labels = labels.long().cuda()

        dis_sim = dis_sim - (1 - labels) * 1e12
        dis_sim = torch.cat((torch.zeros(1).to(dis_sim.device), dis_sim.view(-1)), dim=0)
        loss = torch.logsumexp(dis_sim.view(-1), dim=0)

        return sen1, sen2, loss

    def predict(self, x):
        # print(x)
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)
        return self.extract(x)
