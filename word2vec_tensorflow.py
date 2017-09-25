# -*- coding: utf-8 -*-
import numpy as np
import json
import string
from glob import glob
from collections import Counter, OrderedDict

import tensorflow as tf

def load_data():

    # 維基百科標點符號 常用的標點符號 中華民國教育部, exclude "，" for future use
    punctuation_ch = set(u'。？！、；：「」『』（）［］〔〕【】—…－-～‧《》〈〉﹏＿')
    exclude = set(string.punctuation) | set(punctuation_ch)

    data = []
    for f in glob('chinese-poetry/json/poet.tang.*'):
        with open(f) as data_file:
            for record in json.load(data_file):
                p = record['paragraphs']
                for s in p:
                    data += ''.join(list(filter(lambda x: x not in exclude, s))).split(u'，')
    return data

def build_word_dict(data):

    d = ''.join(data)
    counter = Counter(d).most_common(len(d))
    word_dict = OrderedDict(sorted(filter(lambda v: v[1] > 5, counter), reverse=True, key=lambda v: v[1]))

    return word_dict

def generate_label_data(data, word_list, window_size):

    label_data = []

    for s in data:
        s_buffer = [word_list.index(w) for w in list(filter(lambda x: x in word_list, s))]

        for i in range(len(s_buffer)):
            for c in range(-window_size,  window_size + 1):

                if c == 0 or (i + c) < 0 or (i + c) > (len(s_buffer) - 1):
                    continue

                label_data.append([s_buffer[i],s_buffer[i + c]])

    return np.array(label_data)

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

def get_top(word_list, model, w, top_n):
    i = word_list.index(w)
    dot_product = np.dot(model, model[i].reshape(-1))
    norm = np.linalg.norm(model, axis=1)
    result = dot_product / (norm*norm[i])

    return [(word_list[i], result[i]) for i in result.argsort()[::-1][:top_n + 1]]

# index = 0
# H = 128 # number of hidden layer neurons
# window_size = 2
# negative_sample = 10 # Number of negative examples to sample.
# learning_rate = 0.025
# num_steps = 50001
# batch_size = 128

index = 0
embedding_size = 128  # Dimension of the embedding vector.
window_size = 2
num_sampled = 10    # Number of negative examples to sample.
learning_rate = 0.025
num_steps = 100001
batch_size = 128

data = load_data()

word_dict = build_word_dict(data)
word_list = list(word_dict.keys())

# D = len(word_list) # input dimensionality
vocabulary_size = len(word_list)

print('Preprocess data ...')
label_data = generate_label_data(data, word_list, window_size)

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
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            label_data, batch_size
        )

        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

    final_embeddings = normalized_embeddings.eval()

print('Finish training')

print('-----')
for i, v in get_top(word_list, final_embeddings, u"井", 5):
    print(i, v)

print('-----')
for i, v in get_top(word_list, final_embeddings, u"雲", 5):
    print(i, v)

print('-----')
for i, v in get_top(word_list, final_embeddings, u"峰", 5):
    print(i, v)
