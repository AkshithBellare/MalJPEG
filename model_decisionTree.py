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

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score                
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

model_output=dtc.predict(x_test)   

print(model_output)
print(y_test)

print("Accuracy = %.4f "%(accuracy_score(y_test, model_output,normalize=True)*100))
print("F1 Score = %.4f "%f1_score(y_test, model_output,average='weighted') )
print("Recall Score = %.4f "%recall_score(y_test, model_output, average='weighted') )
