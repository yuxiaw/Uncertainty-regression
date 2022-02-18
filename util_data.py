# SBERT original evaluation reproduction
# ./examples/evaluation/evaluation_stsbenchmark.py 
# In this file, there is a way to evaluate result by running:
# python evaluation_stsbenchmark.py stsb-bert-base
# result is showiing below:
# 2021-04-26 22:47:17 - Evaluation the model on  dataset:
# 2021-04-26 22:48:02 - Cosine-Similarity :	Pearson: 0.8419	Spearman: 0.8505
# 2021-04-26 22:48:02 - Manhattan-Distance:	Pearson: 0.8485	Spearman: 0.8513
# 2021-04-26 22:48:02 - Euclidean-Distance:	Pearson: 0.8485	Spearman: 0.8514
# 2021-04-26 22:48:02 - Dot-Product-Similarity:	Pearson: 0.8026	Spearman: 0.7996

import os
import csv
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import (paired_cosine_distances, paired_euclidean_distances, 
                                     paired_manhattan_distances)
from scipy.stats import pearsonr, spearmanr

def read_csv(input_file, quotechar=None):
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
    return lines

def get_labels(filename, index):
    """ read labels from filename, index is the position index of label in the line """
    lines = read_csv(filename)
    labels = [float(l[index]) for l in lines]
    return labels

def save_array(filename, embeddings):
    # save embeddings into file
    with open(filename, 'wb') as f:
        np.save(f, embeddings)

def load_array(filename):
    with open(filename, 'rb') as f:
        a = np.load(f)
    return a

def embedding_similarity(embeddings1, embeddings2, labels):
    """ embeddings1, embeddings2: list of first and second sentences embeddings
    labels: gold labels, so after cosine similarity between embedding1 and embedding2, 
    their result can be compared with labels by correlation.
    """
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)
    
    print(eval_pearson_cosine, eval_spearman_cosine)
    print(eval_pearson_manhattan, eval_spearman_manhattan)
    print(eval_pearson_euclidean, eval_spearman_euclidean)
    print(eval_pearson_dot, eval_spearman_dot)


def embed2torch(e1, e2, labels):
    e1 = torch.tensor(e1, dtype=torch.float)
    e2 = torch.tensor(e2, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.float)

    x_data = torch.cat((e1, e2, torch.abs(e1-e2), e1*e2), 1)
    y_data = labels
    print(x_data.size(), y_data.size())
    return x_data, y_data


def Totorch(e, labels):
    x_data = torch.tensor(e, dtype=torch.float)
    y_data = torch.tensor(labels, dtype=torch.float)
    print(x_data.size(), y_data.size())
    return x_data, y_data


def eval_correlation(labels, y_pred):
    eval_pearson, _ = pearsonr(labels, y_pred)
    eval_spearman, _ = spearmanr(labels, y_pred)
    print("pearson correlation is %.4f and spearman correlation is %.4f" % (eval_pearson, eval_spearman))
    return eval_pearson, eval_spearman

dataset_path = {
    "stsb": "./dataset/stsbenchmark/sts-train.csv",
    "stsg": "./dataset/sts-g/train.txt",
    "medsts": "./dataset/MedSTS/train.txt",
    "n2c2": "./dataset/N2C2/train.txt",
    "biosses": "./dataset/BIOSSES/train.txt",
    "ebmsass": "./dataset/EBMSASS/txt/train.txt",
    "climedsts": "./dataset/MedSTS/train.txt",
    "clin2c2": "./dataset/N2C2/train.txt", 
    "bioebmsass": "./dataset/EBMSASS/txt/train.txt",
    "yelp": "./dataset/yelp/train.txt",
    "yelpstsb": "./dataset/yelp/train.txt",
}

npy_path = {
    "stsb": "./npy/stsbenchmark/stsb-train.npy",
    "stsg": "./npy/sts-g/train.npy",
    "medsts": "./npy/MedSTS/train.npy",
    "n2c2": "./npy/N2C2/train.npy",
    "biosses": "./npy/BIOSSES/train.npy",
    "ebmsass": "./npy/EBMSASS/train.npy",
    "climedsts": "./npy/medsts/train.npy", # sbert embedding are continue to fine on train
    "clin2c2": "./npy/n2c2sts/train.npy",  # cli# and bio#. these three fined with train.txt
    "bioebmsass": "./npy/ebmsass/train.npy",
    "yelp": "./npy/yelp-nli/train.npy",
    "yelpstsb": "./npy/yelp-stsb/train.npy",
}

index_list = {
    "stsb": [4,5,6],
    "stsg": [2,0,1],
    "medsts": [2,0,1],
    "n2c2": [2,0,1],
    "biosses": [2,0,1],
    "ebmsass": [2,0,1],
    "climedsts": [2,0,1],
    "clin2c2": [2,0,1],
    "bioebmsass": [2,0,1],
}