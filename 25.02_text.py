import os
import re
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from string import punctuation


from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.externals import joblib

import scipy
from scipy.sparse import hstack

data = pd.read_csv('./sentiment_analysis/sentiment_analysis.csv', encoding='latin1', usecols=['Sentiment', 'SentimentText'])
data.columns = ['sentiment', 'text']
data = data.sample(frac=1, random_state=42)
print(data.shape)

#убираем пунктуацию
def tokenize(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r"#(\w+)", '', tweet)
    tweet = re.sub(r"@(\w+)", '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip().lower()
    tokens = word_tokenize(tweet)
    return tokens
#import nltk
#nltk.download('punkt')
tqdm.pandas()

data['tokens'] = data.text.progress_map(tokenize)
data['cleaned_text'] = data['tokens'].map(lambda tokens: ' '.join(tokens))
data[['sentiment', 'cleaned_text']].to_csv('./data/cleaned_text.csv')

data = pd.read_csv('./data/cleaned_text.csv')
print(data.shape)

data.head()

x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'],
                                                    data['sentiment'],
                                                    test_size=0.1,
                                                    random_state=42,
                                                    stratify=data['sentiment'])

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
pd.DataFrame(y_test).to_csv('./data/y_true.csv', index=False, encoding='utf-8')

vectorizer_word = TfidfVectorizer(max_features=40000,
                             min_df=5,
                             max_df=0.5,
                             analyzer='word',
                             stop_words='english',
                             ngram_range=(1, 2))

vectorizer_word.fit(x_train.values.astype('U'))

tfidf_matrix_word_train = vectorizer_word.transform(x_train.values.astype('U'))
tfidf_matrix_word_test = vectorizer_word.transform(x_test.values.astype('U'))

lr_word = LogisticRegression(solver='sag', verbose=2)
lr_word.fit(tfidf_matrix_word_train, y_train)

joblib.dump(lr_word, './data/lr_word_ngram.pkl')

y_pred_word = lr_word.predict(tfidf_matrix_word_test)
pd.DataFrame(y_pred_word, columns=['y_pred']).to_csv('./data/lr_word_ngram.csv', index=False)

y_pred_word = pd.read_csv('./data/lr_word_ngram.csv')
print(accuracy_score(y_test, y_pred_word))

# новая модель - слова в фразы фраз
vectorizer_char = TfidfVectorizer(max_features=40000,
                             min_df=5,
                             max_df=0.5,
                             analyzer='char',
                             ngram_range=(1, 4))

vectorizer_char.fit(x_train.values.astype('U'))

tfidf_matrix_char_train = vectorizer_char.transform(x_train.values.astype('U'))
tfidf_matrix_char_test = vectorizer_char.transform(x_test.values.astype('U'))

lr_char = LogisticRegression(solver='sag', verbose=2)
lr_char.fit(tfidf_matrix_char_train, y_train)

y_pred_char = lr_char.predict(tfidf_matrix_char_test)
joblib.dump(lr_char, './data/lr_char_ngram.pkl')

pd.DataFrame(y_pred_char, columns=['y_pred']).to_csv('./data/lr_char_ngram.csv', index=False)

y_pred_word = pd.read_csv('./data/lr_char_ngram.csv')
print(accuracy_score(y_test, y_pred_word))

# новая модель - обьединение несколько фраз
tfidf_matrix_word_char_train = hstack((tfidf_matrix_word_train, tfidf_matrix_char_train))
tfidf_matrix_word_char_test = hstack((tfidf_matrix_word_test, tfidf_matrix_char_test))

lr_word_char = LogisticRegression(solver='sag', verbose=2)
lr_word_char.fit(tfidf_matrix_word_char_train, y_train)

y_pred_word_char = lr_word_char.predict(tfidf_matrix_word_char_test)
joblib.dump(lr_word_char, './data/lr_word_char_ngram.pkl')

pd.DataFrame(y_pred_word_char, columns=['y_pred']).to_csv('./data/lr_word_char_ngram.csv', index=False)

y_pred_word_char = pd.read_csv('./data/lr_word_char_ngram.csv')
print(accuracy_score(y_test, y_pred_word_char))