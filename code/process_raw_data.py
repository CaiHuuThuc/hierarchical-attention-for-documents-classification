import re
import nltk
import numpy as np 
import pandas as pd
train_dev_ratio = 0.8

def read_train_tsv():
    data = []
    with open('../raw_data/_train.tsv') as f:
        for line in f.readlines():
            data.append(line.strip())
    return np.array(data)

def train_dev_split():
    data = read_train_tsv()
    num_sample = data.shape[0]
    train_idx = np.random.choice(num_sample, int(train_dev_ratio*num_sample), replace=False)
    dev_idx = list(set(range(num_sample)) - set(train_idx))
    with open('../raw_data/train.tsv', 'w') as f:
        for idx in train_idx:
            d = data[idx]
            f.write("%s\n" % (d))
    with open('../raw_data/dev.tsv', 'w') as f:
        for idx in dev_idx:
            d = data[idx]
            f.write("%s\n" % (d))

if __name__ == '__main__':
    
    # labels, texts = read_train_tsv()
    train_dev_split()
    
    # count()
            