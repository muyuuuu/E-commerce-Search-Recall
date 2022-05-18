import torch
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self, model, output_dim=128) -> None:
        super(TextBackbone, self).__init__()

        self.extractor = model
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.extractor.extract(input_ids,
                                     token_type_ids=token_type_ids,
                                     attention_mask=attention_mask)

        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)

        out = self.extractor.extract(**x)
        x = self.fc(out)
        x = F.normalize(x, p=2, dim=-1)
        return x
