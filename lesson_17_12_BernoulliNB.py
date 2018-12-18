import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def gaussian(value, mu, sigma):
    res = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(value-mu)**2/(2*sigma**2))
    return res

feature_list = [
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"]

def runner(features):
    # Importing dataset
    data = pd.read_csv("train.csv")

    # Convert categorical variable to numeric
    data["Sex_cleaned"] = np.where(data["Sex"] == "male", 0, 1)
    data["Embarked_cleaned"] = np.where(data["Embarked"] == "S", 0,
                                        np.where(data["Embarked"] == "C", 1, np.where(data["Embarked"] == "Q", 2, 3)))
    # Cleaning dataset of NaN
    data = data[[
        "Survived",
        "Pclass",
        "Sex_cleaned",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked_cleaned"
    ]].dropna(axis=0, how='any')
    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, test_size=0.3)

    gnb1 = GaussianNB()
    used_features =[
        features
    ]
    # Train classifier
    gnb1.fit(
        X_train[used_features].values,
        X_train["Survived"]
    )
    y_pred1 = gnb1.predict(X_test[used_features])
    print(y_pred1)
    gnb2 = GaussianNB()
    used_features = [
        features
    ]
    # Train classifier
    gnb2.fit(
        X_train[used_features].values,
        X_train["Survived"]
    )
    y_pred2 = gnb2.predict(X_test[used_features])
    print(y_pred2)

    y_pred = y_pred1
    for i in range(len(y_pred)):
        if y_pred1[i] == y_pred2[i]:
            y_pred[i] = y_pred1[i]
        else: y_pred[i] = 1

    return 100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])

def cycle(features):
    y = 0
    for x in range(1, 11):
        y += runner(features)
    print(y/10)

for features in feature_list:
    cycle(features)