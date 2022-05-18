import random


data = []
all = [i for i in range(1, 1001500 + 1)]
with open("/home/20031211375/tf-simcse/simcse-tf2-master/data/qrels.train.tsv", 'r') as f:
    lines = f.readlines()
    for line in lines:
        tmp = []
        line = line.strip().split('\t')
        tmp.append(line[0])
        tmp.append(line[1])
        neg = random.sample(all, 10)
        neg = "#".join([str(i) for i in neg])
        tmp.append(neg)
        data.append("\t".join(tmp))

with open('random.out', 'w') as f:
    for i in data:
        f.writelines(i)
        f.write("\n")

