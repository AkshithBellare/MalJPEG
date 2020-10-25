import pandas as pd
import numpy as np
import glob 

path_benine="/home/axebell/Desktop/theOne/year3CourseMaterials/se303/project/benign_features.csv" #update the paths here 
path_malacious="/home/axebell/Desktop/theOne/year3CourseMaterials/se303/project/malicious_features.csv"

df_benine =pd.read_csv(path_benine,header=None) 
df_benine=df_benine.iloc[1:,]
train_benine, test_benine = np.split(df_benine.sample(frac=1), [int(.75*len(df_benine))]) 

df_malacious =pd.read_csv(path_malacious,header=None) 
df_malacious=df_malacious.iloc[1:,]
train_malacious, test_malacious = np.split(df_malacious.sample(frac=1), [int(.75*len(df_malacious))]) 


train=np.concatenate((train_benine,train_malacious),axis=0)
test=np.concatenate((test_benine,test_malacious),axis=0)

pd.DataFrame(train).to_csv("train.csv",header=None,index=None)
pd.DataFrame(test).to_csv("test.csv",header=None,index=None)



