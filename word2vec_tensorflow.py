# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from time import time
from six.moves import xrange

import tensorflow as tf

def load_saved_data():
    with open("./data/word_dict.json", "r") as f:
        word_dict = dict(json.loads("".join(f.readlines())))

    with open("./data/word_list.json", "r") as f:
        word_list = list(json.loads("".join(f.readlines())))

    with open("./data/label_data.json", "r") as f:
        label_data = np.array(json.loads("".join(f.readlines())))

    return word_dict, word_list, label_data

def generate_batch(label_data, size):
    global index

    if (index + size) > len(label_data):
        d = np.concatenate((label_data[index:index + size], label_data[:(size - len(label_data) + index)]))
        index = size - len(label_data) + index

    else:
        d = label_data[index:index + size]
        index += size

    batch = d[:, 0]
    labels = d[:, 1].reshape(-1, 1)

    return batch, labels

if __name__ == '__main__':
    index = 0
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    num_sampled = 10   # Number of negative examples to sample.
    learning_rate = 1
    num_steps = 500001
    model_name = "./model/w2v_tensorflow.json"

    print('Loading data ...')
    word_dict, word_list, label_data = load_saved_data()

    vocabulary_size = len(word_list)

    start = time()

    graph = tf.Graph()

    with graph.as_default():

        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / np.sqrt(embedding_size))
            )
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size)
            )

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

        init = tf.global_variables_initializer()


    with tf.Session(graph=graph) as session:
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                label_data, batch_size
            )
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    print('Average loss at step ', step, ': ', average_loss)
                    print(index)
                    average_loss = 0

        final_embeddings = normalized_embeddings.eval()

    f = open(model_name, "w")
    f.write(json.dumps(final_embeddings.tolist(), indent=2))
    f.close()

    print('Spend: {0:.2f} min'.format((time() - start)/60))
