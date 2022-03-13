import os
import torch
from torch.utils.data import DataLoader
from data import TEXTDATA, TESTDATA
from model import TextBackbone
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from scipy import stats


def train(epoch, dataloader, model, optimizer, schedular, criterion, log_file):
    model.train()
    all_loss = []
    for idx, (sen1, sen2, label) in enumerate(dataloader):
        _, _, loss = model(sen1, sen2, label)
        all_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedular.step()
        if idx % 500 == 499:
            with open(log_file, 'a+') as f:
                ll = sum(all_loss) / len(all_loss)
                info = str(epoch) + " ==== " + str(idx) + " ==== " + str(
                    ll) + "\n"
                f.write(info)
                all_loss = []


def evaluate(eval_dataloader):
    all_a_vecs = []
    all_b_vecs = []
    all_labels = []
    model.eval()
    for idx, (sen1, sen2, label) in enumerate(eval_dataloader):
        with torch.no_grad():
            sen1, sen2, _ = model(sen1, sen2, label)
            all_a_vecs.append(sen1.cpu().numpy())
            all_b_vecs.append(sen2.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_labels = np.array(all_labels)
    a = all_labels.shape[0]
    all_a_vecs = np.array(all_a_vecs).reshape(a, -1)
    all_b_vecs = np.array(all_b_vecs).reshape(a, -1)
    sims = (all_a_vecs * all_b_vecs).sum(axis=1)
    corrcoef = stats.spearmanr(all_labels, sims).correlation
    return corrcoef


def prepare():
    os.makedirs("./output", exist_ok=True)
    now = datetime.now()
    log_file = now.strftime("%Y_%m_%d_%H_%M_%S") + "_log.txt"
    return "./output/" + log_file


if __name__ == "__main__":
    epochs = 5
    lr = 1e-5
    batch_size = 16
    dataset = TEXTDATA()
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_set, eval_set = torch.utils.data.random_split(
        dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True)
    eval_dataloader = DataLoader(eval_set,
                                 batch_size=batch_size,
                                 shuffle=True)

    model = TextBackbone().cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    num_train_steps = int(len(train_dataloader) * epochs)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps)
    criterion = None

    log_file = prepare()

    for epoch in range(1, epochs + 1):
        train(epoch, train_dataloader, model, optimizer, schedular, criterion,
              log_file)
        torch.save(model.state_dict(), "./output/epoch_{}.pt".format(epoch))
        corr = evaluate(eval_dataloader)
        with open(log_file, 'a+') as f:
            info = str(epoch) + " ==== " + str(corr) + "\n"
            f.write(info)

    if os.path.exists("doc_embedding1"):
        os.remove("doc_embedding1")
    if os.path.exists("query_embedding1"):
        os.remove("query_embedding1")
    if os.path.exists("doc_embedding"):
        os.remove("doc_embedding")
    if os.path.exists("query_embedding"):
        os.remove("query_embedding")

    model.eval()
    testdata = TESTDATA(certain="corpus.tsv")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in testloader:
        with torch.no_grad():
            y = model.predict(x)[0].detach().cpu().numpy().tolist()
            y = [str(i) for i in y]
            info = idx[0] + "\t"
            info = info + ",".join(y)
            with open("doc_embedding1", 'a+') as f:
                f.write(info + "\n")

    testdata = TESTDATA(certain="dev.query.txt")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in testloader:
        with torch.no_grad():
            y = model.predict(x)[0].detach().cpu().numpy().tolist()
            y = [str(i) for i in y]
            info = idx[0] + "\t"
            info = info + ",".join(y)
            with open("query_embedding1", 'a+') as f:
                f.write(info + "\n")
