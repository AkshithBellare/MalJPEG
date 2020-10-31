import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

print('Loading data...')
# load or create your dataset
# df_train = pd.read_csv('dataset/benign_features.csv', header=None)
# df_test = pd.read_csv('dataset/malicious_features.csv', header=None)

# y_train = df_train[0]
# y_test = df_test[0]
# X_train = df_train.drop(0, axis=1)
# X_test = df_test.drop(0, axis=1)
train = pd.read_csv('dataset/benign_features.csv',header=None)
X_train = np.array(train.iloc[:, 0:9])
X_train = X_train[1:]       
y_train = np.array(train.iloc[:,10])
y_train = y_train[1:]

# print("\n", len(X_train), "  ", len(y_train), "\n")

test = pd.read_csv('dataset/malicious_features.csv',header=None)
X_test = np.array(test.iloc[:, 0:9])   
X_test = X_test[1:]       
y_test = np.array(test.iloc[:,10])
y_test = y_test[1:]

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')
from sklearn.metrics import accuracy_score                
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
# print("Accuracy = %.4f ",(accuracy_score(y_test, y_pred,normalize=True)*100))
# print("F1 Score = %.4f ",f1_score(y_test, y_pred,average='weighted') )
# print("Recall Score = %.4f ",recall_score(y_test, y_pred, average='weighted') )
