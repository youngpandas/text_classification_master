from textrnn.config import TEMP_PATH
from setting import ROOT

import gensim
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot
from seaborn import distplot
from collections import Counter
import joblib
import numpy as np

stopword = set(stopwords.words('english'))
google = gensim.models.KeyedVectors.load_word2vec_format(ROOT + '/source/Google-vectors-negative.bin', binary=True)
google_set = set(list(google.vocab))


def get_data(cate):
    if cate == 'all':
        l1, t1 = get_data('train')
        l2, t2 = get_data('test')
        return l1 + l2, t1 + t2
    data = fetch_20newsgroups(subset=cate)
    targets, texts = list(data.target), data.data
    return targets, texts


def handle_text(text: str):
    return text.split()


def handle_text_list(texts):
    res = []
    for i, text in enumerate(texts):
        i += 1
        if i % 1000 == 0:
            print("handle: {:.2f}%  {}/{}".format(i * 100 / len(texts), i, len(texts)))
        res.append(handle_text(text))
    return res


def padding_input(inp, padding_idx, padding_len):
    if len(inp) > padding_len: return inp[:padding_len]
    return inp + [padding_idx] * (padding_len - len(inp))


def padding_input_list(inputs, padding_idx, padding_len):
    res = []
    for i, inp in enumerate(inputs):
        i += 1
        if i % 1000 == 0:
            print("handle: {:.2f}%  {}/{}".format(i * 100 / len(inputs), i, len(inputs)))
        res.append(padding_input(inp, padding_idx, padding_len))
    return res


def get_word_count(word2count, rate):
    word2count = [[w, c] for w, c in word2count.items() if w not in stopword]
    word2count = sorted(word2count, key=lambda x: x[1], reverse=True)
    tol_num = sum([c for w, c in word2count])
    cur_num = 0
    for i, (w, c) in enumerate(word2count):
        cur_num += c
        if cur_num > rate * tol_num:
            return word2count[:i]
    return word2count


def get_padding_len(inputs, rate):
    lens = [len(e) for e in inputs]
    distplot([e for e in lens if e < 1000])
    pyplot.show()
    return sorted(lens)[int(rate * len(inputs))]


def handle_dataset(cate, word2index, padding_len):
    targets, texts = get_data(cate)
    texts = handle_text_list(texts)
    inputs = [[word2index[w] for w in words if w in word2index] for words in texts]
    print(get_padding_len(inputs, 0.9))
    inputs = padding_input_list(inputs, len(word2index), padding_len)
    return targets, inputs


if __name__ == '__main__':
    targets, texts = get_data('all')
    num_classes = len(set(targets))

    word_count = [[w, c] for w, c in sorted(list(Counter(" ".join(texts).split()).items()),
                                            reverse=True, key=lambda x: x[1]) if c > 7][5:]
    word_set = {w for w, c in word_count}
    word2index = {w: i for i, w in enumerate(word_set)}
    index2vec = {i: google[w] if w in google_set else list(np.random.random(300)) for w, i in word2index.items()}
    vectors = [index2vec[i] for i in range(len(index2vec))] + [[0.] * 300]

    inputs = [[w for w in text.split() if w in word_set] for text in texts]
    padding_len = get_padding_len(inputs, 0.9)
    padding_idx = len(word2index)

    y_train, x_train = handle_dataset('train', word2index, padding_len)
    y_test, x_test = handle_dataset('test', word2index, padding_len)

    joblib.dump(word2index, TEMP_PATH + "/len={} w_num={} l_num={}.pkl".format(padding_len, padding_idx, num_classes))
    joblib.dump(word2index, TEMP_PATH + "/word2index.pkl")
    np.save(TEMP_PATH + "/train.input.npy", x_train)
    np.save(TEMP_PATH + "/train.target.npy", y_train)
    np.save(TEMP_PATH + "/test.input.npy", x_test)
    np.save(TEMP_PATH + "/test.target.npy", y_test)
    np.save(TEMP_PATH + "/word2vector.npy", vectors)
