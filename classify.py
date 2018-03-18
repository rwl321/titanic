import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb

def tic_preprocessing(tic_data):
    ids = tic_data['PassengerId'].values
    #train_y = tic_data['Survived'].values
    sample_size = len(ids)

    #Pclass
    p_class = tic_data['Pclass'].values.reshape(sample_size,1)
    x = np.array(p_class)

    #Sex
    le = preprocessing.LabelEncoder()
    sex = le.fit(tic_data['Sex'].values).transform(tic_data['Sex'].values)
    sex = sex.reshape(len(sex),1)
    x = np.append(x, sex, axis=1)
    enc = preprocessing.OneHotEncoder()
    enc.fit(sex)
    x = np.append(x, enc.transform(sex).toarray(), axis=1)

    #Age
    age = tic_data['Age'].values.reshape(sample_size, 1)
    x = np.append(x, age, axis=1)

    #SibSp && Parch
    sibsp = tic_data['SibSp'].values.reshape(sample_size, 1)
    parch = tic_data['Parch'].values.reshape(sample_size, 1)
    x = np.append(x, sibsp, axis=1)
    x = np.append(x, parch, axis=1)

    #Fare
    fare = tic_data['Fare'].values.reshape(sample_size, 1)
    x = np.append(x, fare, axis=1)

    #Embarked
    #embarked= le.fit(tic_data['Embarked'].values).transform(tic_data['Embarked'].values)
    #enc.fit(embarked)
    x = np.append(x, enc.transform(embarked).toarray(), axis=1)


    return ids, x

tic_data = pd.read_csv("data/train.csv")
tic_test_data = pd.read_csv("data/test.csv")

train_y = tic_data['Survived'].values
train_ids, train_x = tic_preprocessing(tic_data)
test_ids, test_x = tic_preprocessing(tic_data)
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)
print train_x.shape
param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 20
#bst = xgb.train(param, dtrain, num_round)
cv = xgb.cv(param, dtrain, 5000, nfold=10, early_stopping_rounds=20, verbose_eval=1)

