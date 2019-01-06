import tensorflow as tf
import numpy as np
import shelve
from datahelper import batch_iter, process
from time import time

def attention(inputs, variable_scopename, double_hidden_size=100, attention_size=50, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    with tf.variable_scope(variable_scopename):
        # Trainable parameters
        w_omega = tf.get_variable(name='w_omega', initializer=tf.constant_initializer(np.random.normal(scale=0.1)), shape=[double_hidden_size, attention_size])
        b_omega = tf.get_variable(name='b_omega', initializer=tf.constant_initializer(np.random.normal(scale=0.1)), shape=[attention_size])
        u_omega = tf.get_variable(name='u_omega', initializer=tf.constant_initializer(np.random.normal(scale=0.1)), shape=[attention_size])

        
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega, name='v')

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

if __name__ == '__main__':
    HIDDEN_SIZE_GRU = 30
    ATTENTION_SIZE = 30
    NUM_CLASSES = 2
    with shelve.open('../data_feed_model/data') as f:
        documents, labels, vocabs, lookup_table, map_word_id = f['data']

    documents_ph = tf.placeholder(shape=[None, None, None], dtype= tf.int32)
    label_ph = tf.placeholder(shape=[None], dtype=tf.int64)
    num_sent_per_document_ph = tf.placeholder(shape=[None], dtype= tf.int32)
    num_word_per_sent_ph = tf.placeholder(shape=[None], dtype= tf.int32)
    max_num_sent_in_a_document_ph = tf.placeholder(dtype= tf.int32)
    max_num_word_in_a_sentence_ph = tf.placeholder(dtype= tf.int32)
    
    with tf.device("/cpu:0"), tf.variable_scope('word-embedding'):
        W_embedding = tf.Variable(name='word-embedding', initial_value= lookup_table, dtype=tf.float32, trainable=True)
        vectors = tf.nn.embedding_lookup(W_embedding, documents_ph, name='vectors')
        vectors = tf.reshape(vectors, shape=[-1, max_num_word_in_a_sentence_ph, 300])

    with tf.device("/cpu:0"), tf.variable_scope("word-encoder"):
        we_cell_fw = tf.contrib.rnn.GRUCell(HIDDEN_SIZE_GRU)
        we_cell_bw = tf.contrib.rnn.GRUCell(HIDDEN_SIZE_GRU)
        (we_output_fw, we_output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                we_cell_fw, we_cell_bw, vectors, \
                                sequence_length=num_word_per_sent_ph, dtype=tf.float32)

        we_output = tf.concat([we_output_fw, we_output_bw], axis=-1)

        
    with tf.device("/cpu:0"):
        wa_output, wa_alphas = attention(we_output, "word-attention", double_hidden_size=2*HIDDEN_SIZE_GRU, attention_size=ATTENTION_SIZE, return_alphas=True)
        wa_output = tf.reshape(wa_output, shape=[-1, max_num_sent_in_a_document_ph, 2*HIDDEN_SIZE_GRU])
        
    with tf.device("/cpu:0"), tf.variable_scope("sentence-encoder"):
        se_cell_fw = tf.contrib.rnn.GRUCell(HIDDEN_SIZE_GRU)
        se_cell_bw = tf.contrib.rnn.GRUCell(HIDDEN_SIZE_GRU)
        (se_output_fw, se_output_bw), _ = tf.nn.bidirectional_dynamic_rnn( \
                                se_cell_fw, se_cell_bw, wa_output, \
                                sequence_length=num_sent_per_document_ph, dtype=tf.float32)

        se_output = tf.concat([se_output_fw, se_output_bw], axis=-1)

    with tf.device("/cpu:0"):
        sa_output, sa_alphas = attention(se_output, "sentence-attention", double_hidden_size=2*HIDDEN_SIZE_GRU, attention_size=ATTENTION_SIZE, return_alphas=True)
        sa_output = tf.reshape(sa_output, shape=[-1, 2*HIDDEN_SIZE_GRU])
    
    with tf.device("/cpu:0"), tf.variable_scope('projection'):
        W = tf.get_variable(initializer=tf.constant_initializer(np.random.normal(scale=1)), shape=[2*HIDDEN_SIZE_GRU, NUM_CLASSES], name="W")
        b = tf.get_variable(initializer=tf.constant_initializer(np.random.normal(scale=1)), shape=[NUM_CLASSES], name="b")
    
        scores = tf.nn.xw_plus_b(sa_output, W, b, name="output")
        predictions = tf.argmax(scores, 1, name="predictions")
    with tf.device("/cpu:0"), tf.name_scope('losses'):
        
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(label_ph, depth=NUM_CLASSES), logits=scores)
        loss = tf.reduce_mean(losses)
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, label_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    with tf.device("/cpu:0"), tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
        training_op = optimizer.minimize(loss)

    # config = tf.ConfigProto(allow_soft_placement = True)

    batch_size = 10
    num_epochs = 10
    n_batches = int(documents.shape[0]//batch_size) + 1
    timer = time()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        batches = batch_iter(documents, labels, batch_size=batch_size, num_epochs=num_epochs)
    
        for step, batch in enumerate(batches):
            documents_in_batch, labels_in_batch, num_sent_in_doc, num_word_whole_sent, max_num_sent_in_a_document, max_num_word_in_a_sentence = batch

            feed_dict = {
                documents_ph: documents_in_batch,
                label_ph: labels_in_batch,
                num_sent_per_document_ph: num_sent_in_doc,
                num_word_per_sent_ph: num_word_whole_sent,
                max_num_sent_in_a_document_ph: max_num_sent_in_a_document,
                max_num_word_in_a_sentence_ph: max_num_word_in_a_sentence
            }
            loss_, _ = sess.run([loss, training_op], feed_dict=feed_dict)
            
            if (step + 1) % n_batches == 0 or step > n_batches*num_epochs - 1:
                    epoch = (step + 1) // n_batches
                    print("Epoch: %d Loss: %f Took %fs" % (epoch, loss_, time() - timer))
                    saver.save(sess, '../saved_model/model.ckpt', global_step=step)
                    timer = time()
                        