import gensim
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import numpy as np

stopword = set(stopwords.words('english'))
google = gensim.models.KeyedVectors.load_word2vec_format('./source/GoogleNews-vectors-negative300.bin', binary=True)
google_set = set(list(google.vocab))


def get_data(cate):
    if cate == 'all':
        l1, t1 = get_data('train')
        l2, t2 = get_data('test')
        return l1 + l2, t1 + t2
    data = fetch_20newsgroups(subset=cate)
    targets, texts = list(data.target), data.data
    return targets, texts


targets, texts = get_data('all')
word_count = list(Counter(" ".join(texts).split()).items())
word_count = sorted(word_count, reverse=True, key=lambda x: x[1])
word_count = [[w, c] for w, c in word_count if c > 7][5:]
word_set = {w for w, c in word_count}
word2index = {w: i for i, (w, c) in enumerate(word_set)}
index2vec = {i: google[w] if w in google_set else list(np.random.random(300)) for w, i in word2index.items()}
vectors = [index2vec[i] for i in range(len(index2vec))]
vectors.append([0.] * 300)
