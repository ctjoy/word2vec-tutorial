# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json

def get_top(word_list, model, w, top_n):
    i = word_list.index(w)
    dot_product = np.dot(model, model[i].reshape(-1))
    norm = np.linalg.norm(model, axis=1)
    result = dot_product / (norm*norm[i])

    return [(word_list[i], result[i]) for i in result.argsort()[::-1][:top_n + 1]]

def get_calculated_top(word_list, model, w1, w2, w3, top_n):
    wid1, wid2, wid3 = word_list.index(w1), word_list.index(w2), word_list.index(w3)
    v1, v2, v3 = model[wid1], model[wid2], model[wid3]
    vec = v1 + (v2 - v3)

    dot_product = np.dot(model, vec.reshape(-1))
    vec_norm = np.linalg.norm(vec)
    norm = np.linalg.norm(model, axis=1)
    result = dot_product / (norm*vec_norm)

    print('{} + {} - {}'.format(w1, w2, w3))
    return [(word_list[i], result[i]) for i in result.argsort()[::-1][:top_n + 2] if i not in [wid1, wid2, wid3]]

if __name__ == '__main__':

    # with open("./model/w2v_tensorflow.json", "r") as f:
    with open("./model/w2v.json", "r") as f:
        model = np.array(json.loads("".join(f.readlines())))

    with open("./data/word_list.json", "r") as f:
        word_list = list(json.loads("".join(f.readlines())))

    print('-----')
    for i, v in get_top(word_list, model, u"井", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"雲", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"峰", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"風", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"母", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"女", 5):
        print(i, v)

    print('-----')
    for i, v in get_calculated_top(word_list, model, u"女", u"父", u"男", 5):
        print(i, v)
