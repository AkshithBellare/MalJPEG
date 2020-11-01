import pandas as pd
import numpy as np
import glob 

path_benign="/home/axebell/Desktop/MalJPEG/dataset/benign_features.csv" #update the paths here 
path_malicious="/home/axebell/Desktop/MalJPEG/dataset/malicious_features.csv"

df_benign =pd.read_csv(path_benign,header=None) 
df_benign=df_benign.iloc[1:,]
train_benign, test_benign = np.split(df_benign.sample(frac=1), [int(.75*len(df_benign))]) 

df_malicious =pd.read_csv(path_malicious,header=None) 
df_malicious=df_malicious.iloc[1:,]
train_malicious, test_malicious = np.split(df_malicious.sample(frac=1), [int(.75*len(df_malicious))]) 


train=np.concatenate((train_benign,train_malicious),axis=0)
test=np.concatenate((test_benign,test_malicious),axis=0)

pd.DataFrame(train).to_csv("train.csv",header=None,index=None)
pd.DataFrame(test).to_csv("test.csv",header=None,index=None)



