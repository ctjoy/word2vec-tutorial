# -*- coding: utf-8 -*-
import numpy as np
import json
import string
from glob import glob
from time import time
from collections import Counter, OrderedDict

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

if __name__ == '__main__':
    print('Process data ...')
    start = time()

    window_size = 2
    data = load_data()

    word_dict = build_word_dict(data)
    word_list = list(word_dict.keys())

    total_words = float(np.sum(list(word_dict.values())))

    label_data = generate_label_data(data, word_list, window_size, total_words)

    with open('./data/word_dict.json', "w") as f:
        f.write(json.dumps(word_dict, indent=2))
    print('Save word_dict ...')

    with open('./data/word_list.json', "w") as f:
        f.write(json.dumps(word_list, indent=2))
    print('Save word_list ...')

    with open('./data/label_data.json', "w") as f:
        f.write(json.dumps(label_data.tolist(), indent=2))
    print('Save label_data ...')

    print('Spend: {0:.2f} min'.format((time() - start)/60))
