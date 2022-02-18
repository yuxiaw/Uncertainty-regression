import os
# from sentence_transformers import SentenceTransformer
from util_data import dataset_path, npy_path, index_list
from util_data import (read_csv, save_array, load_array, get_labels, embedding_similarity)

def get_bert_sentence_embedding(sentences, model_save_path = 'stsb-bert-base'):
    model = SentenceTransformer(model_save_path)

    #Sentences are encoded by calling model.encode()
    sentence_embeddings = model.encode(sentences)

    print(len(sentence_embeddings))
    print(type(sentence_embeddings[0]))
    print(sentence_embeddings[0].shape)
    
    return sentence_embeddings

def get_sbert_embedding(cached_npy=None, filename=None, index_order=[4,5,6]):
    if os.path.exists(cached_npy):
        arr = load_array(cached_npy)
        n = len(arr) // 2
        embeddings1, embeddings2 = arr[:n], arr[n:]
    else:
        lines = read_csv(filename)
        n = len(lines)
        print(n, lines[0])
        i, j, k = index_order # index of score, first and second sentence in each line, sequentially
        labels = [float(l[i]) for l in lines]
        S1 = [l[j] for l in lines]
        S2 = [l[k] for l in lines]

        #Our sentences we like to encode
        assert(len(S1) == len(S2))
        assert(len(S1) == len(labels))
        sentences = S1 + S2
        sentence_embeddings = get_bert_sentence_embedding(sentences)
        embeddings1, embeddings2 = sentence_embeddings[:n], sentence_embeddings[n:]
        if cached_npy:
            save_array(cached_npy, sentence_embeddings)
    return embeddings1, embeddings2

def get_single_sent_embedding(cached_npy=None, filename=None, model_save_path='bert-base-nli-mean-tokens'):
    if os.path.exists(cached_npy):
        embeddings = load_array(cached_npy)
        lines = read_csv(filename)
        labels = [float(l[-1]) for l in lines]
    else:
        lines = read_csv(filename)
        n = len(lines)
        print(n, lines[0])
        labels = [float(l[-1]) for l in lines]
        S = [l[0] for l in lines]
        assert(len(S) == len(labels))
        embeddings = get_bert_sentence_embedding(S)
        if cached_npy:
            save_array(cached_npy, embeddings)
    print(embeddings.shape, len(labels))
    return (embeddings, labels)

def load_single_sent_features(dataset_name, returnFL=True):
    return_list = []

    train_path = dataset_path[dataset_name]
    train_npy = npy_path[dataset_name]

    dev_path = train_path[:-9]+"dev"+train_path[-4:]
    dev_npy = train_npy[:-9]+"dev.npy"

    test_path = train_path[:-9]+"test"+train_path[-4:]
    test_npy = train_npy[:-9]+"test.npy"

    train_e, train_labels = get_single_sent_embedding(cached_npy=train_npy, filename=train_path)
    return_list.append((train_e, train_labels))
    print("train embedding done!")

    if os.path.exists(dev_path):
        dev_e, dev_labels = get_single_sent_embedding(cached_npy=dev_npy, filename=dev_path)
        return_list.append((dev_e, dev_labels))
        print("dev embedding done!")
    if os.path.exists(test_path):
        test_e, test_labels = get_single_sent_embedding(cached_npy=test_npy, filename=test_path)                                 
        return_list.append((test_e, test_labels))
        print("test embedding done!")
    
    if returnFL:
        return return_list
    
def save_sbert_features(dataset_name, returnFL=True):
    return_list = []

    index_order = index_list[dataset_name]
    label_index = index_order[0]

    train_path = dataset_path[dataset_name]
    train_npy = npy_path[dataset_name]

    dev_path = train_path[:-9]+"dev"+train_path[-4:]
    dev_npy = train_npy[:-9]+"dev.npy"

    test_path = train_path[:-9]+"test"+train_path[-4:]
    test_npy = train_npy[:-9]+"test.npy"
    
    train_labels = get_labels(train_path, label_index)
    train_e1, train_e2 = get_sbert_embedding(cached_npy=train_npy, 
                                             filename=train_path,
                                             index_order=index_order)
    return_list.append((train_e1, train_e2, train_labels))
    print("train embedding done!")

    if os.path.exists(dev_path):
        dev_labels = get_labels(dev_path, label_index)
        dev_e1, dev_e2 = get_sbert_embedding(cached_npy=dev_npy, 
                                            filename=dev_path,
                                            index_order=index_order)
        return_list.append((dev_e1, dev_e2, dev_labels))
        print("dev embedding done!")
    if os.path.exists(test_path):
        test_labels = get_labels(test_path, label_index)
        test_e1, test_e2 = get_sbert_embedding(cached_npy=test_npy, 
                                            filename=test_path,
                                            index_order=index_order)
        return_list.append((test_e1, test_e2, test_labels))
        print("test embedding done!")
    
    if returnFL:
        return return_list

if __name__ == "__main__":
    test_labels = get_labels("./dataset/stsbenchmark/sts-test.csv", 4)
    test_e1, test_e2 = get_sbert_embedding(cached_npy="stsb-test.npy", 
                                           filename="./dataset/stsbenchmark/sts-test.csv")
    embedding_similarity(test_e1, test_e2, test_labels)
    # output result as below, they are same as previous code result, proving embedding is reproduced
    # 0.8419134988160621 0.8505240524335684
    # 0.8485120149141858 0.8512783172883697
    # 0.8485197034928713 0.85135885099808
    # 0.8025811992439804 0.7995967633794774

    train_labels = get_labels("./dataset/stsbenchmark/sts-train.csv", 4)
    train_e1, train_e2 = get_sbert_embedding(cached_npy="stsb-train.npy", 
                                         filename="./dataset/stsbenchmark/sts-train.csv",
                                         index_order = [4,5,6])
    dev_labels = get_labels("./dataset/stsbenchmark/sts-dev.csv", 4)
    dev_e1, dev_e2 = get_sbert_embedding(cached_npy="stsb-dev.npy", 
                                         filename="./dataset/stsbenchmark/sts-dev.csv",
                                         index_order = [4,5,6])
    embedding_similarity(dev_e1, dev_e2, dev_labels)

    # load training data
    data_list = save_sbert_features(dataset_name="stsb", returnFL=True)
    train_e1, train_e2, train_labels = data_list[0]
    dev_e1, dev_e2, dev_labels = data_list[1]
    test_e1, test_e2, test_labels = data_list[2]





