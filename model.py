from operator import ne
from re import X
import torch
from transformers import AutoModel


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='./pretrained/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        self.fc = torch.nn.Linear(768, output_dim)
        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, anchor, pos, neg):
        anchor["input_ids"] = anchor["input_ids"].squeeze()
        anchor["attention_mask"] = anchor["attention_mask"].squeeze()
        anchor["token_type_ids"] = anchor["token_type_ids"].squeeze()
        pos["input_ids"] = pos["input_ids"].squeeze()
        pos["attention_mask"] = pos["attention_mask"].squeeze()
        pos["token_type_ids"] = pos["token_type_ids"].squeeze()
        anchor_out = self.fc(self.extractor(**anchor).pooler_output)
        pos_out = self.fc(self.extractor(**pos).pooler_output)
        neg_out = torch.empty(0, 128).to(anchor_out.device)
        for i in neg:
            i["input_ids"] = i["input_ids"].squeeze()
            i["attention_mask"] = i["attention_mask"].squeeze()
            i["token_type_ids"] = i["token_type_ids"].squeeze()
            output_i = self.fc(self.extractor(**i).pooler_output)
            output_i = self.norm(output_i)
            neg_out = torch.cat((neg_out, output_i), dim=0)

        anchor_out = self.norm(anchor_out)
        pos_out = self.norm(pos_out)

        return anchor_out, pos_out, neg_out

    def predict(self, x):
        # print(x)
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)
        return self.norm(self.fc(self.extractor(**x).pooler_output))
