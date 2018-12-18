import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

twenty_train.data[0]
#перечень классов
twenty_train.target_names
print(twenty_train.target_names)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

print(X_train_tfidf.shape)

#text_lr = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
#                    ('lr', LogisticRegression(multi_class='multinomial',solver ='newton-cg'))])
#stop_words='english') - убирает все возможные ненужные словар использую Английскуий словарь типа a, the , an , and
#с ним процент 0,83
# без 0,82

text_lr = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(multi_class='multinomial',solver ='newton-cg'))])


_ = text_lr.fit(twenty_train.data, twenty_train.target)

predicted = text_lr.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))