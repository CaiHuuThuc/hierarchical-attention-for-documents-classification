import os
import csv
import numpy as np
import shelve

import numpy as np
import re
from time import time
from math import sqrt
import tensorflow as tf
import sys
from lxml import etree
import nltk
from datahelper import generate_lookup_word_embedding

def encode_sentences_for_testing(documents, map_word_id):
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

def read_files_for_testing(filename='../raw_data/dev.tsv'):
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

def update_lookup_table_for_testing(documents, lookup_table, map_word_id):

    n_out_of_vocabs = 0
    out_of_vocabs = list()
    old_n_vocabs = len(map_word_id.keys())
    oov_map_word_id = dict()
    
    for doc in documents:
        for sent in doc:
            for word in sent:
                if word not in map_word_id and word not in oov_map_word_id:
                    w = word
                    oov_map_word_id[w] = n_out_of_vocabs
                    n_out_of_vocabs += 1
                    out_of_vocabs.append(w)

    #update map_word_id
    for idx, w in enumerate(out_of_vocabs):
        map_word_id[w] = idx + old_n_vocabs
        
    #update lookup table
    dims = lookup_table.shape[1]
    updated_lookup_table = np.zeros(shape=[lookup_table.shape[0] + n_out_of_vocabs, dims])

    oov_lookup_table = generate_lookup_word_embedding(out_of_vocabs, oov_map_word_id)
    for idx in range(lookup_table.shape[0]):
        updated_lookup_table[idx, :] = lookup_table[idx, :]

    for idx in range(oov_lookup_table.shape[0]):
        updated_lookup_table[idx + old_n_vocabs, :] = oov_lookup_table[idx, :]

    return updated_lookup_table


def get_feed_dict_for_testting(documents, updated_lookup_table, map_word_id):

    dims = updated_lookup_table.shape[1]

    encoded_test_documents = encode_sentences_for_testing(documents, map_word_id)

    for idx in range(documents.shape[0]):
        encoded_test_doc = encoded_test_documents[idx]
        max_num_sent_in_a_document = encoded_test_doc.shape[0]

        num_sent_per_document = np.array([encoded_test_doc.shape[0]])

        num_word_per_sent = []
        for sent in encoded_test_doc:
            num_word_per_sent.append(sent.shape[0])

        max_num_word_in_a_sentence = max(num_word_per_sent)

        vectors = np.zeros(shape=(1, max_num_sent_in_a_document, max_num_word_in_a_sentence, dims), dtype=np.float32)
        

        for subidx, sent in enumerate(documents[idx]):
            for subsubidx, word in enumerate(documents[idx][subidx]):
                id_of_word = map_word_id[word]
                vectors[0, subidx, subsubidx, :] = updated_lookup_table[id_of_word, :]
        
        
        

        feed_dict = {
                "vectors": vectors,
                "num_sent_per_document_ph" : num_sent_per_document,
                "num_word_per_sent_ph" : num_word_per_sent,
                "max_num_sent_in_a_document_ph" : max_num_sent_in_a_document,
                "max_num_word_in_a_sentence_ph" : max_num_word_in_a_sentence
                }

        yield feed_dict



if __name__ == '__main__':
    type_embeddings = sys.argv[1].strip()
    tagging = sys.argv[2].strip()
    type_w_emb = sys.argv[3].strip()
    phrase = sys.argv[4].strip()


    assert phrase in ['dev', 'test', 'debug']
    assert job in ['prepare-data', 'tsv-to-xml', 'stat']
    print()
    print("Type embeddings: %s " % type_embeddings)
    print("Tagging: %s" % tagging)
    print('Variant word embedding: %s' % type_w_emb)
    print('Phrase: %s' % phrase)
    print('Job: %s' % job)

    thong_ke(phrase)


