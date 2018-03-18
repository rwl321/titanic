
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=20)

train_df = train_df.drop(["Ticket", 'Cabin'], axis=1)
test_df = test_df.drop(["Ticket", 'Cabin'], axis=1)
combine = [train_df, test_df]


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = { "Mr": 1, "Miss":2, "Mrs":3, "Master":4, "Rare": 5}
for dataset in combine:
    dataset["Title"] = dataset['Title'].map(title_mapping)
    dataset["Title"] = dataset['Title'].fillna(0)

for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({'male':0, 'female':1}).astype(int)

