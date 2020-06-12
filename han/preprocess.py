from han.config import TEMP_PATH

from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from matplotlib import pyplot
from seaborn import distplot
from collections import Counter
import joblib
import numpy as np

end_set = set('.,?!;:')
stopword = set(stopwords.words('english'))
word_map = {
    "i'll": "i will",
    "it'll": "it will",
    "we'll": "we will",
    "he'll": "he will",
    "they'll": "they will",
    "i'd": "i would",
    "we'd": "we would",
    "he'd": "he would",
    "they'd": "they would",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "here's": "here is",
    "there's": "there is",
    "we're": "we are",
    "they're": "they are",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "wanna": "want to",
    "can't": "can not",
    "ain't": "are not",
    "isn't": "is not",
}


def handle_text(text: str):
    words = " . ".join([line for line in text.lower().split("\n") if line != ""]).split()
    res1 = []
    for w in words:
        if '-' in w:
            res1 += w.split('-')
            continue
        if w.isalpha():
            res1.append(w)
            continue
        if w.count('.') <= 1:
            res1.append(w)
            continue
    res1 = [w for w in res1 if w != ""]

    res2 = []
    for w in res1:
        post = ""
        if w[-1] in end_set:
            post = w[-1]
            w = w[:-1]
        w = "".join([c for c in w if c == "'" or c.isalpha() or c.isalnum()])
        res2.append(w)
        res2.append(post)
    res2 = [w for w in res2 if w != ""]

    res3 = []
    for w in res2:
        if w in word_map:
            res3 += word_map[w].split()
        else:
            res3.append(w)
    return [w for w in res3 if w != ""]


def handle_text_list(texts):
    res = []
    for i, text in enumerate(texts):
        i += 1
        if i % 1000 == 0:
            print("handle: {:.2f}%  {}/{}".format(i * 100 / len(texts), i, len(texts)))
        res.append(handle_text(text))
    return res


def get_word_count(word2count, rate):
    word2count = [[w, c] for w, c in word2count.items() if w not in stopword and w.isalpha()]
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


def get_data(cate):
    if cate == 'all':
        l1, t1 = get_data('train')
        l2, t2 = get_data('test')
        return l1 + l2, t1 + t2
    data = fetch_20newsgroups(subset=cate)
    targets, texts = list(data.target), data.data
    return targets, texts


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

    texts = handle_text_list(texts)
    word2count = Counter([w for words in texts for w in words])

    word_count = get_word_count(word2count, 0.95)

    print("单词量:", len(word_count))

    word2index = {w: i for i, (w, c) in enumerate(word_count)}
    inputs = [[word2index[w] for w in words if w in word2index] for words in texts]
    padding_len = get_padding_len(inputs, 0.90)
    print("补全长度:", padding_len)

    padding_idx = len(word2index)

    y_train, x_train = handle_dataset('train', word2index, padding_len)
    y_test, x_test = handle_dataset('test', word2index, padding_len)

    joblib.dump(word2index, TEMP_PATH + "/w2i.pkl")
    joblib.dump(word2index, TEMP_PATH + "/len={} w_num={} l_num={}.pkl".format(padding_len, padding_idx, num_classes))
    np.save(TEMP_PATH + "/train.x.npy", x_train)
    np.save(TEMP_PATH + "/train.y.npy", y_train)
    np.save(TEMP_PATH + "/test.x.npy", x_test)
    np.save(TEMP_PATH + "/test.y.npy", y_test)
