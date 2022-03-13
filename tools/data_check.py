def data_check(file, file_type="doc"):
    """ check if a file is UTF8 without BOM,
        doc_embedding index starts with 0,
        query_embedding index starts with 200001,
        the dimension of the embedding is 128.
    """
    count = 0
    with open(file) as f:
        for line in f:
            sp_line = line.strip().split("\t")
            if len(sp_line) != 2:
                print("[Error] Please check your line. The line should be two parts, i.e. index \t embedding")
                break
            index, embedding = sp_line
            if count == 0:
                if file_type == "doc" and index != "1":
                    print("[Error] The index of doc_embedding is not 1. Please check it.")
                    print("line: ", sp_line)
                    break
                elif file_type == "query" and index != "200001":
                    print("[Error] The index of query_embedding is not 200001. Please check it.")
                    print("line: ", sp_line)
                    break
            if len(embedding.split(",")) != 128:
                print("[Error] Please check the dimension of embedding. The dimension is not 128. The line number is {}".format(count+1))
                break
            count += 1
    print("Check done!\n")


if __name__ == "__main__":
    print("*"*10, "Checking doc_embedding ...")
    data_check("doc_embedding", file_type="doc")
    print("*"*10, "Checking query_embedding ...")
    data_check("query_embedding", file_type="query")

