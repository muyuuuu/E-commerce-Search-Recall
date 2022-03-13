import os
import torch
from torch.utils.data import DataLoader
from data import TEXTDATA, TESTDATA
from loss import triplet_loss
from model import TextBackbone
from datetime import datetime


def train(dataloader, model, optimizer, schedular, criterion, log_file):
    model.train()
    for idx, (anchor, pos, neg) in enumerate(dataloader):
        # anchor["input_ids"] = anchor["input_ids"].squeeze()
        # pos["input_ids"] = pos["input_ids"].squeeze()
        # for i in neg:
        #     i["input_ids"] = i["input_ids"].squeeze()
        anchor, pos, neg = model(anchor, pos, neg)
        # pos = model(pos)
        # neg = model(neg)
        optimizer.zero_grad()
        loss = criterion(anchor, pos, neg)
        loss.backward()
        optimizer.step()
        schedular.step()
        if idx % 100 == 99:
            with open(log_file, 'a+') as f:
                info = str(idx) + " ==== " + str(loss) + "\n"
                f.write(info)
            break


def prepare():
    os.makedirs("./output", exist_ok=True)
    now = datetime.now()
    log_file = now.strftime("%Y_%m_%d_%H_%M_%S") + "_log.txt"
    return "./output/" + log_file


if __name__ == "__main__":
    dataset = TEXTDATA()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    model = TextBackbone().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=5e-4)
    epochs = 1
    schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                           T_0=10,
                                                           T_mult=2)
    criterion = triplet_loss

    log_file = prepare()

    for epoch in range(1, epochs + 1):
        train(dataloader, model, optimizer, schedular, criterion, log_file)
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       "./output/epoch_{}.pt".format(epoch))

    if os.path.exists("doc_embedding"):
        os.remove("doc_embedding")
    if os.path.exists("query_embedding"):
        os.remove("query_embedding")

    model.eval()
    testdata = TESTDATA(certain="corpus.tsv")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in testloader:
        y = model.predict(x)[0].detach().cpu().numpy().tolist()
        y = [str(i) for i in y]
        info = idx[0] + "\t"
        info = info + "\t".join(y)
        with open("doc_embedding", 'a+') as f:
            f.write(info + "\n")

    testdata = TESTDATA(certain="dev.query.txt")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in testloader:
        y = model.predict(x)[0].detach().cpu().numpy().tolist()
        y = [str(i) for i in y]
        info = idx[0] + "\t"
        info = info + "\t".join(y)
        with open("query_embedding", 'a+') as f:
            f.write(info + "\n")
