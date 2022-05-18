import os
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import AdamW, get_linear_schedule_with_warmup
import utils_unilm
from data import Supervised, TESTDATA
from model import TextBackbone
import modeling_unilm, configuration_unilm, tokenization_unilm


def prepare_log_file():
    os.makedirs("./output", exist_ok=True)
    now = datetime.now()
    log_file = now.strftime("%Y_%m_%d_%H_%M_%S") + "_log.txt"
    return "./output/" + log_file


def unilm_pretrain():

    config = configuration_unilm.UnilmConfig.from_pretrained(
        "./torch_unilm_model")
    tokenizer = tokenization_unilm.UnilmTokenizer.from_pretrained(
        "./torch_unilm_model")
    model = modeling_unilm.UnilmForSeq2Seq.from_pretrained(
        "./torch_unilm_model", config=config).cuda()

    max_pred = 20
    mask_prob = 0.2
    # 最大长度，防止丢失信息
    max_seq_length = 120
    skipgram_prb = 0.3
    skipgram_size = 2
    mask_whole_word = False
    data_tokenizer = tokenizer
    unilm_pipeline = [
        # 1 / 3
        # 0.7 被 mask
        utils_unilm.Preprocess4Seq2seq(max_pred,
                                       mask_prob + 0.5,
                                       list(tokenizer.vocab.keys()),
                                       tokenizer.convert_tokens_to_ids,
                                       max_len=max_seq_length,
                                       skipgram_prb=0,
                                       skipgram_size=0,
                                       mask_whole_word=mask_whole_word,
                                       mask_source_words=False,
                                       tokenizer=data_tokenizer),
        # 1 / 3
        utils_unilm.Preprocess4BiLM(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len=max_seq_length,
                                    skipgram_prb=skipgram_prb,
                                    skipgram_size=skipgram_size,
                                    mask_whole_word=False,
                                    mask_source_words=True,
                                    tokenizer=data_tokenizer),
        # 1 / 6
        utils_unilm.Preprocess4RightLM(max_pred,
                                       mask_prob,
                                       list(tokenizer.vocab.keys()),
                                       tokenizer.convert_tokens_to_ids,
                                       max_len=max_seq_length,
                                       skipgram_prb=skipgram_prb,
                                       skipgram_size=skipgram_size,
                                       mask_whole_word=False,
                                       mask_source_words=True,
                                       tokenizer=data_tokenizer),
        # 1 / 6
        utils_unilm.Preprocess4LeftLM(max_pred,
                                      mask_prob,
                                      list(tokenizer.vocab.keys()),
                                      tokenizer.convert_tokens_to_ids,
                                      max_len=max_seq_length,
                                      skipgram_prb=skipgram_prb,
                                      skipgram_size=skipgram_size,
                                      mask_whole_word=False,
                                      mask_source_words=True,
                                      tokenizer=data_tokenizer),
    ]

    trainset = utils_unilm.Seq2SeqDataset(file="./data/corpus.tsv",
                                          batch_size=64,
                                          tokenizer=tokenizer,
                                          max_len=115,
                                          short_sampling_prob=0.1,
                                          sent_reverse_order=False,
                                          bi_uni_pipeline=unilm_pipeline)

    train_dataloader = DataLoader(
        trainset,
        batch_size=64,
        collate_fn=utils_unilm.batch_list_to_batch_tensors,
        pin_memory=False)

    epochs = 30
    max_grad_norm = 1
    num_train_steps = int(len(train_dataloader) * epochs)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps)
    log_file = prepare_log_file()
    all_loss = []

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [t.cuda() if t is not None else None for t in batch]
            input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
            masked_lm_loss = model(input_ids,
                                   segment_ids,
                                   input_mask,
                                   lm_label_ids,
                                   masked_pos=masked_pos,
                                   masked_weights=masked_weights)
            loss = masked_lm_loss
            all_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            schedular.step()

            if step % 3000 == 2999:
                with open(log_file, 'a+') as f:
                    avg_loss = round(sum(all_loss) / len(all_loss), 6)
                    f.write(" [Pretrain] Epoch: {}, Step: {}, Loss is: {} \n".
                            format(epoch, step + 1, avg_loss))
                    all_Loss = []

        if epoch > 0 and epoch % 5 == 0:
            output_model_file = "./output/pretrain_{}_model.pt".format(epoch)
            output_optim_file = "./output/pretrain_{}_optim.pt".format(epoch)
            output_sched_file = "./output/pretrain_{}_sched.pt".format(epoch)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(optimizer.state_dict(), output_optim_file)
            torch.save(schedular.state_dict(), output_sched_file)

    return log_file, model, data_tokenizer


def unsup_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device='cuda')
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1),
                                       y_pred.unsqueeze(0),
                                       dim=2)

    similarities = similarities - torch.eye(y_pred.shape[0],
                                            device='cuda') * 1e12

    similarities = similarities / lamda

    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, model, optimizer, schedular, criterion, log_file, epoch):
    num = 2
    model.train()
    all_loss = []
    for idx, data in enumerate(dataloader):
        input_ids = data['input_ids'].view(len(data['input_ids']) * num,
                                           -1).cuda()
        attention_mask = data['attention_mask'].view(
            len(data['attention_mask']) * num, -1).cuda()
        token_type_ids = data['token_type_ids'].view(
            len(data['token_type_ids']) * num, -1).cuda()
        pred = model(input_ids, attention_mask, token_type_ids)
        optimizer.zero_grad()
        loss = criterion(pred)
        all_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        schedular.step()
        if idx % 500 == 499:
            with open(log_file, 'a+') as f:
                t = round(sum(all_loss) / len(all_loss), 6)
                info = " [Finetune] Epoch: {}, Step: {}, Loss: {}\n".format(
                    epoch, idx + 1, t)
                f.write(info)
                all_loss = []


if __name__ == "__main__":
    # _, _, _ = unilm_pretrain()
    log_file = prepare_log_file()
    model_name_or_path = "./torch_unilm_model"

    config = configuration_unilm.UnilmConfig.from_pretrained(
        model_name_or_path)
    data_tokenizer = tokenization_unilm.UnilmTokenizer.from_pretrained(
        model_name_or_path)
    model = modeling_unilm.UnilmForSeq2Seq.from_pretrained(
        model_name_or_path, config=config).cuda()

    model.load_state_dict(torch.load("./output/pretrain_100_model.pt"))
    # 已经加载参数
    model = TextBackbone(model).cuda()

    dataset = Supervised(data_tokenizer=data_tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=60,
                            shuffle=True,
                            drop_last=False)

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
    epochs = 15
    num_train_steps = int(len(dataloader) * epochs)
    schedular = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_steps,
        num_training_steps=num_train_steps)

    criterion = unsup_loss

    for epoch in range(1, epochs + 1):
        train(dataloader, model, optimizer, schedular, criterion, log_file,
              epoch)
        torch.save(model.state_dict(),
                   "./output/sup_epoch_{}.pt".format(epoch))

    if os.path.exists("doc_embedding"):
        os.remove("doc_embedding")
    if os.path.exists("query_embedding"):
        os.remove("query_embedding")

    model.eval()

    testdata = TESTDATA(data_tokenizer=data_tokenizer, certain="dev.query.txt")
    testloader = DataLoader(testdata, batch_size=1, shuffle=False)
    for idx, x in testloader:
        with torch.no_grad():
            y = model.predict(x)[0].detach().cpu().numpy().tolist()
            y = [str(i) for i in y]
            info = idx[0] + "\t"
            info = info + ",".join(y)
            with open("query_embedding", 'a+') as f:
                f.write(info + "\n")

    testdata = TESTDATA(data_tokenizer=data_tokenizer, certain="corpus.tsv")
    testloader = DataLoader(testdata, batch_size=60, shuffle=False)
    for idx, x in testloader:
        with torch.no_grad():
            y = model.predict(x).detach().cpu().numpy().tolist()
            for x1, y1 in zip(idx, y):
                y1 = [str(round(i, 8)) for i in y1]
                info = x1 + "\t"
                info = info + ",".join(y1)
                with open("doc_embedding", 'a+') as f:
                    f.write(info + "\n")
