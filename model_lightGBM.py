import pandas as pd
import numpy as np
from sklearn import utils 
from glob import glob

train = pd.read_csv('./dataset/train.csv',header=None)
x_train = np.array(train.iloc[:, 0:9])        
y_train = np.array(train.iloc[:,10])

test = pd.read_csv('./dataset/test.csv',header=None)
x_test = np.array(test.iloc[:, 0:9])        
y_test = np.array(test.iloc[:,10])

import lightgbm as lgb

lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

from sklearn.metrics import accuracy_score                
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

model_output = gbm.predict(x_test, num_iteration=gbm.best_iteration)

print(model_output)
print(y_test)

print("Accuracy = %.4f "%(accuracy_score(y_test, model_output.round(),normalize=True)*100))
print("F1 Score = %.4f "%f1_score(y_test, model_output,average='weighted') )
print("Recall Score = %.4f "%recall_score(y_test, model_output, average='weighted') )
