# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from time import time
from six.moves import xrange

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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def build_unigram_table(word_dict):
    frq = map(lambda v: int(v**0.75), word_dict.values())
    t = [np.full(v, i, dtype=int) for i, v in enumerate(frq)]

    return np.concatenate(t) # for nagative sampling

if __name__ == '__main__':

    index = 0
    batch_size = 128
    embedding_size = 128 # number of hidden layer neurons
    num_sampled= 10 # Number of negative examples to sample.
    learning_rate = 0.025
    num_steps = 50001
    model_name = './model/w2v.json'

    print('Loading data ...')
    word_dict, word_list, label_data = load_saved_data()

    vocabulary_size = len(word_list)

    start = time()

    model = {}
    model['W1'] = np.random.randn(vocabulary_size,embedding_size) / np.sqrt(embedding_size) # "Xavier" initialization
    model['W2'] = np.zeros((vocabulary_size,embedding_size))

    unigram_table = build_unigram_table(word_dict)

    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(label_data, batch_size)

        total_err = 0

        for t in range(batch_size):
            err_buffer = np.zeros(embedding_size)
            i = batch_inputs[t]

            for n in range(num_sampled + 1):

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

    f = open(model_name, "w")
    f.write(json.dumps(model['W1'].tolist(), indent=2))
    f.close()

    print('Spend: {0:.2f} min'.format((time() - start)/60))
