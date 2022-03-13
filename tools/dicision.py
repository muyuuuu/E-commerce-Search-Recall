with open("query_embedding1", 'r') as f1:
    with open("query_embedding", 'a+') as f2:
        lines = f1.readlines()
        for line in lines:
            temp = line.strip().split("\t")
            id = str(temp[0])
            data = id + "\t" + ",".join([str(round(float(i), 8)) for i in temp[1].split(",")])
            f2.write(data + "\n")

with open("doc_embedding1", 'r') as f1:
    with open("doc_embedding", 'a+') as f2:
        lines = f1.readlines()
        for line in lines:
            temp = line.strip().split("\t")
            id = str(temp[0])
            data = id + "\t" + ",".join([str(round(float(i), 8)) for i in temp[1].split(",")])
            f2.write(data + "\n")
