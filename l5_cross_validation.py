import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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

def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))

def CV(df, classifier, nfold, norm=True):
    acc = []
    for i in range(nfold):
        y = df['class']
        train, test = stratified_split(y)

        if norm:
            X_train = norm_df(df.iloc[train, 0:8])
            X_test = norm_df(df.iloc[test, 0:8])
        else:
            X_train = df.iloc[train, 0:8]
            X_test = df.iloc[test, 0:8]

        y_train = y[train]
        y_test = y[test]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc.append(accuracy(y_test, y_pred))

    return acc

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url, names=names)

logreg = LogisticRegression()
rf = RandomForestClassifier()

print(np.array(CV(df, logreg, 10)).mean())
print(np.array(CV(df, logreg, 10, norm=False)).mean())
print(np.array(CV(df, rf, 10, norm=True)).mean())
print(np.array(CV(df, rf, 10, norm=False)).mean())

#print(CV(df, logreg, 10, norm=False))
