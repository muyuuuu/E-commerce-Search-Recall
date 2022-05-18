import json

data = []

with open("doc_embedding", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split("\t")
        tmp = {}
        tmp["qid"] = line[0]
        line[2] = line[2].split(',')
        tmp["input_ids"] = [int(i) for i in line[2] if i != "0"]
        tmp["input_ids"] = tmp["input_ids"][:-1]
        data.append(tmp)

with open("corpus.json", 'w') as f:
    for i in data:
        json.dump(i, f)
        f.write("\n")
