import os
import shelve
import numpy as np
import tensorflow as tf
from time import time

from testing_datahelper import read_files_for_testing, get_feed_dict_for_testting, update_lookup_table_for_testing

from math import sqrt
import sys
import re
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def predict():


    with shelve.open('../data_feed_model/data') as f:
        documents, labels, vocabs, lookup_table, map_word_id = f['data']

    with tf.Session() as sess:
        
        saver = tf.train.import_meta_graph('../saved_model/model.ckpt-18512.meta')
    
        with tf.device("/cpu:0"):
            print('Prepare data: Start')
            saver.restore(sess, tf.train.latest_checkpoint('../saved_model/'))
            graph = tf.get_default_graph()


            # writer = tf.summary.FileWriter('../summaries', graph)
     
            lookup_table = sess.run(graph.get_tensor_by_name('word-embedding/word-embedding:0'))

            documents_ph = graph.get_tensor_by_name('Placeholder:0')
            label_ph = graph.get_tensor_by_name('Placeholder_1:0')
            num_sent_per_document_ph = graph.get_tensor_by_name('Placeholder_2:0')
            num_word_per_sent_ph = graph.get_tensor_by_name('Placeholder_3:0')
            max_num_sent_in_a_document_ph = graph.get_tensor_by_name('Placeholder_4:0')
            max_num_word_in_a_sentence_ph = graph.get_tensor_by_name('Placeholder_5:0')

            vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
            predict_op = graph.get_tensor_by_name("projection/predictions:0")
        
            test_labels, test_documents = read_files_for_testing()

            updated_lookup_table = update_lookup_table_for_testing(test_documents, lookup_table, map_word_id)
            print('Prepare data: Done - Start predict')
            total = 0
            true = 0
            feed_dicts = get_feed_dict_for_testting(test_documents, updated_lookup_table, map_word_id)
            for idx, fd in enumerate(feed_dicts):
                feed_dict = {
                        vectors: fd["vectors"],
                        num_sent_per_document_ph : fd["num_sent_per_document_ph"],
                        num_word_per_sent_ph : fd["num_word_per_sent_ph"],
                        max_num_sent_in_a_document_ph : fd["max_num_sent_in_a_document_ph"],
                        max_num_word_in_a_sentence_ph : fd["max_num_word_in_a_sentence_ph"]
                }
                
                
                predict = sess.run(predict_op, feed_dict=feed_dict)
                predict = predict[0]

                # print(predict, end='\t')
                # print(test_labels[idx])
                if predict == test_labels[idx]:
                    true += 1
                total += 1
            print("Total: %d\tTrue: %d" % (total, true))
            print("Accuracy: %f%%" % (true / total * 100))



if __name__ == '__main__':
    predict()
    