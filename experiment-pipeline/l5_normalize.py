import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def stratified_split(y, proportion=0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y) #only unique
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds

def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized

def norm_df(df):
    result = df.copy()

    for feature in df.columns:
        result[feature] = norm_arr(result[feature])

    return result

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)

train, test = stratified_split(df['class'])

# X_train = df.iloc[train, 0:8]
# X_test = df.iloc[test, 0:8]

X_train = norm_df(df.iloc[train, 0:8])
X_test = norm_df(df.iloc[test, 0:8])

y_train = df['class'][train]
y_test = df['class'][test]

# print(X_train.shape)
# print(X_test.shape)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))

print(accuracy(y_test, y_pred))
