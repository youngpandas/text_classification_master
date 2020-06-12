from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

fetch_20newsgroups(
    subset='train',  # 加载那一部分数据集 train/test
    shuffle=True,  # 将数据集随机排序
    remove=(),  # ('headers','footers','quotes') 去除部分文本
    download_if_missing=True  # 如果没有下载过，重新下载
)

newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)
print(vectors.nnz / float(vectors.shape[0]))

# MultinomialNB实现文本分类
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

newsgroups_test = fetch_20newsgroups(subset='test')  # 加载测试集
vectors_test = vectorizer.transform(newsgroups_test.data)  # 提取测试集tfidf特征
clf = MultinomialNB(alpha=0.1)
clf.fit(vectors, newsgroups_train.target)  # 训练
pred = clf.predict(vectors_test)  # 预测
print(f1_score(newsgroups_test.target, pred, average='macro'))
print(accuracy_score(newsgroups_test.target, pred))
