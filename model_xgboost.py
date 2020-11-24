import pandas as pd
import numpy as np
from sklearn import utils 
from glob import glob
import xgboost as xgb

def perf_measure(y_actual, y_hat):
    T = 0
    F = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i]:
            T+=1
        if y_actual[i] != y_hat[i]:
            F+=1
        if y_actual[i] == 0 and y_hat[i] == 100:
            FP += 1
        if y_actual[i] == 100 and y_hat[i] == 100:
            TP += 1
        if y_actual[i] == 0 and y_hat[i] == 0:
            TN += 1
        if y_actual[i] == 100 and y_hat[i] == 0:
            FN += 1
    return (TP, FP,TN, FN)

def main():

    train = pd.read_csv('./dataset/train.csv',header=None)
    x_train = np.array(train.iloc[:, 0:9])        
    y_train = np.array(train.iloc[:,10])

    test = pd.read_csv('./dataset/test.csv',header=None)
    x_test = np.array(test.iloc[:, 0:9])        
    y_test = np.array(test.iloc[:,10])

    from xgboost import XGBClassifier
    gbo = XGBClassifier()
    gbo.fit(x_train,y_train)

    model_output=gbo.predict(x_test)   

    print(model_output)
    print(y_test)

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import plot_roc_curve
    from matplotlib import pyplot as plt

    fpr,tpr, _ = roc_curve(y_test, model_output,pos_label=1)
    plot_roc_curve(gbo,x_test,y_test)
    #plt.show()
    plt.savefig('XGBoost.png')

    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

    TPR = TP / (TP + FN)
    print("The TPR is : %.6f" %TPR)

    FPR = FP/(FP+TN)
    print("The FPR is : %.6f" %FPR)

    IDR = TPR * (1-FPR)
    print("The IDR is : %.6f" %IDR)

    AUC = (roc_auc_score(y_test, model_output))
    print("The AUC value is : %.4f" %AUC)

    # plt.plot(FPR, TPR, linestyle='dotted', label='Decision Tree Model')

    # # axis labels
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # show the legend
    # plt.legend()
    # # show the plot
    # plt.show()

if __name__=='__main__':
    main()