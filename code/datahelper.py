import os
import csv
import numpy as np
import shelve

import numpy as np
import re
from math import sqrt
import nltk

def read_train_tsv(filename='../raw_data/train.tsv'):
    labels, texts = [], []
    with open(filename) as f:
        for line in f.readlines():
            label, text = re.split('\t', line.strip())
            doc = []
            sent_text = nltk.sent_tokenize(text)
            for sentence in sent_text:
                tokens = nltk.word_tokenize(sentence.strip())
                doc.append(tokens)
            labels.append(int(label))
            texts.append(doc)
    return np.array(labels), np.array(texts)

def get_vocabs():
    vocabs = []
    vocabs.append('PAD')
    vocabs.append('UNK')
    _, documents = read_train_tsv()
    for doc in documents:
        for sent in doc:
            for word in sent:
                if word not in vocabs:
                    vocabs.append(word)
    _, documents = read_train_tsv('../raw_data/dev.tsv')
    for doc in documents:
        for sent in doc:
            for word in sent:
                if word not in vocabs:
                    vocabs.append(word)
    
    vocabs.append('')
    return vocabs

def get_map_word_id(vocabs):
    map_word_id = dict()
    for id_, word in enumerate(vocabs):
        map_word_id[word] = id_
    return map_word_id

def generate_lookup_word_embedding(vocabs, map_word_id):
    from gensim.models.keyedvectors import KeyedVectors
    filename = './GoogleNews-vectors-negative300.bin'
    pretrained_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
    dims = 300
    n_vocabs = len(vocabs)

    lookup_table = np.zeros(shape=[n_vocabs, dims])
    for _, word in enumerate(vocabs):
        idx = map_word_id[word]
        if word in pretrained_word2vec:
            lookup_table[idx, :] = pretrained_word2vec[word]
        else:
            lookup_table[idx, :] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)

    return lookup_table

def encode_document(documents, map_word_id):
    encoded_documents = np.zeros(shape=[documents.shape[0]], dtype=object)
    for doc_idx, doc in enumerate(documents):
        max_word_in_a_sentence = max([len(d) for d in doc])
        tmp_sent = np.zeros(shape=[len(doc)], dtype=object)

        for sent_idx, sent in enumerate(doc):
            tmp_word = np.zeros(shape=[len(sent)], dtype=np.int32)
            
            for word_idx, word in enumerate(sent):
                idx_of_word = map_word_id[word]
                tmp_word[word_idx] = idx_of_word
            tmp_sent[sent_idx] = tmp_word
        
        encoded_documents[doc_idx] = tmp_sent
    return encoded_documents

def batch_iter(documents, labels, batch_size=32, num_epochs=50, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    n_samples = documents.shape[0]
    num_batches_per_epoch = int((n_samples-1)/batch_size) + 1
    idx_of_word_pad = 0 #index of 'PAD' word
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(n_samples))

            documents = documents[shuffle_indices]
            labels = labels[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, n_samples)

            max_num_sent_in_a_document = max([len(s) for s in documents[start_index:end_index]])
            max_num_word_in_a_sentence = 0
            for doc in documents[start_index:end_index]:
                for sent in doc:
                    max_num_word_in_a_sentence = max(max_num_word_in_a_sentence, len(sent))


            actual_batch_size = min(batch_size, end_index - start_index)
            padded_documents = np.zeros(shape=[actual_batch_size, max_num_sent_in_a_document, max_num_word_in_a_sentence], dtype=np.int32)

            num_sent_per_document = np.array([len(s) for s in documents[start_index:end_index]])
            num_word_per_sent = []
            for doc in documents[start_index:end_index]:
                t = []
                for sent in doc:
                    t.append(len(sent))
                if len(doc) < max_num_sent_in_a_document:
                    t.extend([0]*(max_num_sent_in_a_document - len(doc)))
                num_word_per_sent.extend(t)

            num_sent_per_document = np.array(num_sent_per_document, dtype=np.int32)
            num_word_per_sent = np.array(num_word_per_sent, dtype=np.int32)
            for idx_document, doc in enumerate(documents[start_index:end_index]):
                for idx_sent, sent in enumerate(doc):
                    for idx_word, word in enumerate(sent):
                        padded_documents[idx_document][idx_sent][idx_word] = word

            yield padded_documents, labels[start_index:end_index], num_sent_per_document, num_word_per_sent, max_num_sent_in_a_document, max_num_word_in_a_sentence

def process():
    labels, texts = read_train_tsv()
    vocabs = get_vocabs()
    map_word_id = get_map_word_id(vocabs)
    lookup_table = generate_lookup_word_embedding(vocabs, map_word_id)
    encoded_documents = encode_document(texts, map_word_id)
    with shelve.open('../data_feed_model/data') as f:
        f['data'] = encoded_documents, labels, vocabs, lookup_table, map_word_id
    # return encoded_documents, labels, vocabs, lookup_table, map_word_id

def thong_ke(pharase):
    filename = "../raw_data/%s.tsv" % phrase

    max_num_sentences_in_a_doc = 0
    max_num_word_in_a_sent = 0
    avg_num_sentences_in_documents = 0
    

    _, texts = read_train_tsv(filename)
    max_num_sentences_in_a_doc = max([len(doc) for doc in texts])
    avg_num_sentences_in_documents = np.array([len(doc) for doc in texts]).mean()

    max_num_word_in_a_sent = max([max([len(sent) for sent in doc]) for doc in texts])

    avg_num_word_in_sentences = []
    for doc in texts:
        for sent in doc:
            avg_num_word_in_sentences.append(len(sent))

    avg_num_word_in_sentences = np.array(avg_num_word_in_sentences).mean()
    print("Phrase: %s" % phrase)
    print("max_num_sentences_in_a_doc: %.2f" % max_num_sentences_in_a_doc)
    print("max_num_word_in_a_sent: %.2f" % max_num_word_in_a_sent)

    print("avg_num_sentences_in_documents: %.2f" % avg_num_sentences_in_documents)
    print("avg_num_word_in_sentences: %.2f" % avg_num_word_in_sentences)
    print("\n\n")
if __name__ == '__main__':

    # _, texts = read_train_tsv()
    # process()

    for phrase in ['train', 'dev']:
        thong_ke(phrase)