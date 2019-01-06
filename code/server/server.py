import grpc
from concurrent import futures
import time
import sys
sys.path.append('../')
import shelve
from live_predict_helper import get_feed_dict
from math import sqrt

# import the generated classes
import calculator_pb2
import calculator_pb2_grpc

import tensorflow as tf
import os

import nltk
nltk.download('punkt')
import numpy as np


def run_model(sentence):
    global map_word_id

    saver = tf.train.import_meta_graph('../../saved_model/model.ckpt-18512.meta')
    sess =  tf.Session()
    with tf.device("/cpu:0"):        
        saver.restore(sess, tf.train.latest_checkpoint('../../saved_model/'))
        graph = tf.get_default_graph()


    with tf.device("/cpu:0"):        
        lookup_table = sess.run(graph.get_tensor_by_name('word-embedding/word-embedding:0'))

        documents_ph = graph.get_tensor_by_name('Placeholder:0')
        label_ph = graph.get_tensor_by_name('Placeholder_1:0')
        num_sent_per_document_ph = graph.get_tensor_by_name('Placeholder_2:0')
        num_word_per_sent_ph = graph.get_tensor_by_name('Placeholder_3:0')
        max_num_sent_in_a_document_ph = graph.get_tensor_by_name('Placeholder_4:0')
        max_num_word_in_a_sentence_ph = graph.get_tensor_by_name('Placeholder_5:0')

        alpha_words = graph.get_tensor_by_name('word-attention/alphas:0')
        alpha_sentences = graph.get_tensor_by_name('sentence-attention/alphas:0')

        vectors = graph.get_tensor_by_name('word-embedding/vectors:0')
        predict_op = graph.get_tensor_by_name("projection/predictions:0")

        doc, fd = get_feed_dict(sentence, lookup_table, map_word_id)
    
        feed_dict = {
                vectors: fd["vectors"],
                num_sent_per_document_ph : fd["num_sent_per_document_ph"],
                num_word_per_sent_ph : fd["num_word_per_sent_ph"],
                max_num_sent_in_a_document_ph : fd["max_num_sent_in_a_document_ph"],
                max_num_word_in_a_sentence_ph : fd["max_num_word_in_a_sentence_ph"]
            }
    
        alpha_w, alpha_s, predict = sess.run([alpha_words, alpha_sentences, predict_op], feed_dict=feed_dict)

    alpha_w = repr(list([list(a) for a in alpha_w]))
    alpha_s = repr(list([list(a) for a in alpha_s]))

    print(alpha_s)
    print("\n\n")
    print(alpha_w)

    return str(doc), str(predict[0]), alpha_s, alpha_w


class CalculatorServicer(calculator_pb2_grpc.CalculatorServicer):

    # calculator.square_root is exposed here
    # the request and response are of the data type
    # calculator_pb2.Number

    def analysis(self, request, context):
        res = calculator_pb2.Res()
        res.token, res.label, res.alpha_sentences, res.alpha_words = run_model(request.sentence)
        return res

if __name__ == '__main__':
    with shelve.open('../../data_feed_model/data') as f:
        documents, labels, vocabs, lookup_table, map_word_id = f['data']

    
        
    # create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # use the generated function `add_CalculatorServicer_to_server`
    # to add the defined class to the server
    calculator_pb2_grpc.add_CalculatorServicer_to_server(
        CalculatorServicer(), server)

    

    # listen on port 50051
    print('Starting server. Listening on port 50051.')
    server.add_insecure_port('0.0.0.0:50051')
    server.start()

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    
            
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)