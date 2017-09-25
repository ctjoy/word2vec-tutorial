# -*- coding: utf-8 -*-
import numpy as np
import json
import string
from glob import glob
from collections import Counter, OrderedDict

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def build_unigram_table(word_dict):
    frq = map(lambda v: int(v**0.75), word_dict.values())
    t = [np.full(v, i, dtype=int) for i, v in enumerate(frq)]

    return np.concatenate(t) # for nagative sampling


def get_keeping_rate(w, word_list, total_words):
    z = word_dict[w] / total_words
    keeping_rate = (np.sqrt(z / 0.001) + 1) * (0.001 / z)
    return keeping_rate # discard the word appears too frequently

def generate_label_data(data, word_list, window_size, total_words):

    check = lambda x: x in word_list and get_keeping_rate(x, word_list, total_words) > np.random.uniform()
    label_data = []

    for s in data:
        s_buffer = [word_list.index(w) for w in list(filter(check, s))]

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
    dot_product = np.dot(model['W1'], model['W1'][i].reshape(-1))
    norm = np.linalg.norm(model['W1'], axis=1)
    result = dot_product / (norm*norm[i])

    return [(word_list[i], result[i]) for i in result.argsort()[::-1][:top_n + 1]]

if __name__ == '__main__':

    index = 0
    H = 128 # number of hidden layer neurons
    window_size = 2
    negative_sample = 10 # Number of negative examples to sample.
    learning_rate = 0.025
    num_steps = 50001
    batch_size = 128

    data = load_data()

    word_dict = build_word_dict(data)
    word_list = list(word_dict.keys())
    total_words = float(np.sum(list(word_dict.values())))

    D = len(word_list) # input dimensionality

    print('Preprocess data ...')
    label_data = generate_label_data(data, word_list, window_size, total_words)

    model = {}
    model['W1'] = np.random.randn(D,H) / np.sqrt(H) # "Xavier" initialization
    model['W2'] = np.zeros((D,H))

    unigram_table = build_unigram_table(word_dict)

    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(label_data, batch_size)

        total_err = 0

        for t in range(batch_size):
            err_buffer = np.zeros(H)
            i = batch_inputs[t]

            for n in range(negative_sample + 1):

                if n == 0: # positive example
                    target = 1
                    o = batch_labels[t][0]
                else: # negative example
                    target = 0
                    o = unigram_table[np.random.randint(len(unigram_table))]

                w1 = model['W1'][i]
                w2 = model['W2'][o]

                err = sigmoid(np.dot(w1, w2)) - target

                err_buffer += err * model['W2'][o]

                model['W2'][o] -= learning_rate * err * model['W1'][i]

                total_err += abs(err)

            model['W1'][i] -= learning_rate * err_buffer

        average_loss += total_err

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

    print('Finish training')

    print('-----')
    for i, v in get_top(word_list, model, u"井", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"雲", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"峰", 5):
        print(i, v)
