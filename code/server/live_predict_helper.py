import nltk
import numpy as np
from math import sqrt

def get_feed_dict(text, lookup_table, map_word_id):
    doc = []
    sent_text = nltk.sent_tokenize(text)
    for sentence in sent_text:
        tokens = nltk.word_tokenize(sentence.strip())
        doc.append(tokens)
    
    num_sent_per_document = np.array([len(doc)])
    max_num_sent_in_a_document = len(doc)
    
    num_word_per_sent = []
    for sent in doc:
        num_word_per_sent.append(len(sent))
    num_word_per_sent = np.array(num_word_per_sent)
    
    max_num_word_in_a_sentence = max(num_word_per_sent)
    
    dims = lookup_table.shape[1]

    vectors = np.zeros(shape=(1, max_num_sent_in_a_document, max_num_word_in_a_sentence, dims), dtype=np.float32)
    for subidx, sent in enumerate(doc):
        for subsubidx, word in enumerate(doc[subidx]):
            if word in map_word_id:
                id_of_word = map_word_id[word]
                vectors[0, subidx, subsubidx, :] = lookup_table[id_of_word, :]
            else:
                vectors[0, subidx, subsubidx, :] = np.random.uniform(-sqrt(3.0/dims), sqrt(3.0/dims), dims)

    feed_dict = {
            "vectors": vectors,
            "num_sent_per_document_ph" : num_sent_per_document,
            "num_word_per_sent_ph" : num_word_per_sent,
            "max_num_sent_in_a_document_ph" : max_num_sent_in_a_document,
            "max_num_word_in_a_sentence_ph" : max_num_word_in_a_sentence
        }
    return doc, feed_dict
