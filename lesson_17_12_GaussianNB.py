# -*- coding: utf-8 -*-
"""
Editor de Spyder
Este es un archivo temporal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Importing dataset
data = pd.read_csv("train.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["embarked"]=="S",0,np.where(data["embarked"]=="C",1,np.where(data["embarked"]=="Q",2,3)))


# Cleaning dataset of NaN
data=data[[
    "survived",
    "pclass",
    "Sex_cleaned",
    "age",
    "sibsp",
    "parch",
    "fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')

X_train, X_test = train_test_split(data, test_size=0.5, random_state=int(time.time()))


# Instantiate the classifier
gnb = GaussianNB()
used_features =[
    "pclass",
    "Sex_cleaned",
    "age",
    "sibsp",
    "parch",
    "fare",
    "Embarked_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["survived"] != y_pred).sum(),
          100*(1-(X_test["survived"] != y_pred).sum()/X_test.shape[0])
))


mean_survival=np.mean(X_train["survived"])
mean_not_survival=1-mean_survival
print("Survival prob = {:03.2f}%, Not survival prob = {:03.2f}%"
      .format(100*mean_survival,100*mean_not_survival))



mean_fare_survived = np.mean(X_train[X_train["survived"]==1]["fare"])
std_fare_survived = np.std(X_train[X_train["survived"]==1]["fare"])
mean_fare_not_survived = np.mean(X_train[X_train["survived"]==0]["fare"])
std_fare_not_survived = np.std(X_train[X_train["survived"]==0]["fare"])

print("mean_fare_survived = {:03.2f}".format(mean_fare_survived))
print("std_fare_survived = {:03.2f}".format(std_fare_survived))
print("mean_fare_not_survived = {:03.2f}".format(mean_fare_not_survived))
print("std_fare_not_survived = {:03.2f}".format(std_fare_not_survived))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
used_features =["fare"]
y_pred = gnb.fit(X_train[used_features].values, X_train["survived"]).predict(X_test[used_features])
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["survived"] != y_pred).sum(),
          100*(1-(X_test["survived"] != y_pred).sum()/X_test.shape[0])
))
print("Std Fare not_survived {:05.2f}".format(np.sqrt(gnb.sigma_)[0][0]))
print("Std Fare survived: {:05.2f}".format(np.sqrt(gnb.sigma_)[1][0]))
print("Mean Fare not_survived {:05.2f}".format(gnb.theta_[0][0]))
print("Mean Fare survived: {:05.2f}".format(gnb.theta_[1][0]))