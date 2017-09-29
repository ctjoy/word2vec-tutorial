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

if __name__ == '__main__':
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
    for i, v in get_top(word_list, model, u"黃", 5):
        print(i, v)

    print('-----')
    for i, v in get_top(word_list, model, u"女", 5):
        print(i, v)
