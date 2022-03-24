import torch
from transformers import AutoModel, BertConfig
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='./pretrained/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        config = BertConfig.from_pretrained(pretrained)
        config.attention_probs_dropout_prob = 0.3
        config.hidden_dropout_prob = 0.3
        self.extractor = AutoModel.from_pretrained(pretrained,
                                                   config=config).cuda()
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.extractor(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             output_hidden_states=True)

        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)

        out = self.extractor(**x, output_hidden_states=True)
        first = out.hidden_states[1].transpose(1, 2)
        last = out.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(
            first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(
            -1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)),
                        dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x
