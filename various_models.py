import pandas as pd
import numpy as np
from sklearn import utils 
from glob import glob
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from matplotlib import pyplot as plt

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

def cal_values(TP, FP, TN, FN, y_test, model_output):

    TPR = TP / (TP + FN)
    print("The TPR is : %.4f" %TPR)

    FPR = FP/(FP+TN)
    print("The FPR is : %.4f" %FPR)

    IDR = TPR * (1-FPR)
    print("The IDR is : %.4f" %IDR)

    AUC = (roc_auc_score(y_test, model_output))
    print("The AUC value is : %.4f" %AUC)

    return (TPR,FPR,IDR,AUC)


def dec_tree(x_train, y_train, x_test, y_test):

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    model_output=dtc.predict(x_test)   

    # print(model_output)
    # print(y_test)

    fpr,tpr, _ = roc_curve(y_test, model_output,pos_label=1)
    # plot_roc_curve(dtc,x_test,y_test)
    # #plt.show()
    # plt.savefig('Decision Tree.png')

    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

    TPR,FPR, IDR, AUC = cal_values(TP,FP,TN,FN,y_test,model_output)

    return (TPR,FPR, IDR, AUC)

def native_bayes(x_train, y_train, x_test, y_test):

    from sklearn.naive_bayes import BernoulliNB
    naive_bayes_model= BernoulliNB()                                 
    naive_bayes_model.fit(x_train, y_train)

    model_output=naive_bayes_model.predict(x_test)   

    # print(model_output)
    # print(y_test)

    fpr,tpr, _ = roc_curve(y_test, model_output,pos_label=1)
    # plot_roc_curve(naive_bayes_model,x_test,y_test)
    # #plt.show()
    # plt.savefig('Native Bayes.png')

    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

    TPR,FPR, IDR, AUC = cal_values(TP,FP,TN,FN,y_test,model_output)

    return (TPR,FPR, IDR, AUC)

def xgboost(x_train, y_train, x_test, y_test):

    from xgboost import XGBClassifier
    gbo = XGBClassifier()
    gbo.fit(x_train,y_train)

    model_output=gbo.predict(x_test)   

    # print(model_output)
    # print(y_test)

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import plot_roc_curve
    from matplotlib import pyplot as plt

    fpr,tpr, _ = roc_curve(y_test, model_output,pos_label=1)
    # plot_roc_curve(gbo,x_test,y_test)
    # #plt.show()
    # plt.savefig('XGBoost.png')

    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

    TPR,FPR, IDR, AUC = cal_values(TP,FP,TN,FN,y_test,model_output)

    return (TPR,FPR, IDR, AUC)

def random_forest(x_train, y_train, x_test, y_test):

    from sklearn.ensemble import RandomForestClassifier 
    rfc = RandomForestClassifier(n_estimators=100)                                 
    rfc.fit(x_train, y_train) 

    model_output=rfc.predict(x_test)   

    # print(model_output)
    # print(y_test)

    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import plot_roc_curve
    from matplotlib import pyplot as plt

    fpr,tpr, _ = roc_curve(y_test, model_output,pos_label=1)
    # plot_roc_curve(rfc,x_test,y_test)
    # #plt.show()
    # plt.savefig('Random Forest.png')

    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=model_output)

    TPR,FPR, IDR, AUC = cal_values(TP,FP,TN,FN,y_test,model_output)

    return (TPR,FPR, IDR, AUC)

def light_gbm(x_train, y_train, x_test, y_test):

    import lightgbm as lgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, roc_curve, auc
    import matplotlib.pyplot as plt

    lgb_train = lgb.Dataset(x_train, label=y_train)
        
    gridParams = {
    'learning_rate': [0.07],
    'n_estimators': [8,16],
    'num_leaves': [20, 24, 27],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], 
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    #'reg_alpha' : [1, 1.2],
    #'reg_lambda' : [ 1.2, 1.4],
    }



    params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.07,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2,
          'reg_lambda': 1.2,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'
          }

    mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 5, 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

    grid = GridSearchCV(mdl, gridParams, verbose=2, cv=4, n_jobs=-1)

    # Run the grid
    grid.fit(x_train, y_train)

    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)


    lgbm = lgb.train(params,
                 lgb_train,
                 280,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )
    predictions_lgbm_prob = lgbm.predict(x_test)
    predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0)

    acc_lgbm = accuracy_score(y_test,predictions_lgbm_01)
    print(acc_lgbm)
    for i in range(len(y_test)):
        if y_test[i] == 100:
            y_test[i] = 1
    
    #plot ROC curve
    plt.figure()
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_lgbm_prob)
    roc_auc = auc(false_positive_rate, recall)
    plt.plot(false_positive_rate, recall, 'b', label = 'LGBMClassifier(AUC = %0.3f)' %roc_auc)
    plt.legend(loc='lower right')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    predictions_lgbm = [ele * 100 for ele in predictions_lgbm_01]
    y_test = [ele * 100 for ele in y_test]
    
    TP, FP, TN, FN = perf_measure(y_actual=y_test, y_hat=predictions_lgbm)

    TPR,FPR, IDR, AUC = cal_values(TP,FP,TN,FN,y_test,predictions_lgbm)

    return (TPR,FPR, IDR, AUC)



def main():

    train = pd.read_csv('./dataset/train.csv',header=None)
    x_train = np.array(train.iloc[:, 0:9])        
    y_train = np.array(train.iloc[:,10])

    test = pd.read_csv('./dataset/test.csv',header=None)
    x_test = np.array(test.iloc[:, 0:9])        
    y_test = np.array(test.iloc[:,10])

    rows,cols = (5,1)           # change to (5,1) for 5 different models
    data = [[0 for i in range(cols)] for j in range(rows)]

    print("Naive Bayes")

    tpr1,fpr1,idr1,auc1 = native_bayes(x_train, y_train, x_test, y_test)
    data[0].append(tpr1)
    data[1].append(fpr1)
    data[2].append(idr1)
    data[3].append(auc1)

    print("Decision Tree")

    tpr2,fpr2,idr2,auc2 = dec_tree(x_train, y_train, x_test, y_test)
    data[0].append(tpr2)
    data[1].append(fpr2)
    data[2].append(idr2)
    data[3].append(auc2)

    print("Random Forest")

    tpr3,fpr3,idr3,auc3 = random_forest(x_train, y_train, x_test, y_test)
    data[0].append(tpr3)
    data[1].append(fpr3)
    data[2].append(idr3)
    data[3].append(auc3)

    print("XGBoost")    

    tpr4 ,fpr4 ,idr4 ,auc4 = xgboost(x_train, y_train, x_test, y_test)
    data[0].append(tpr4)
    data[1].append(fpr4)
    data[2].append(idr4)
    data[3].append(auc4)

    tpr5 ,fpr5 ,idr5 ,auc5 = light_gbm(x_train, y_train, x_test, y_test)
    data[0].append(tpr5)
    data[1].append(fpr5)
    data[2].append(idr5)
    data[3].append(auc5)

    for i in range(5):
        data[i].pop(0)  
 
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(5)     #change to 5 with addition of Light GBM
    bar_width = 0.20
    opacity = 0.8

    # def autolabel(rects):
    #    for rect in rects:
    #     height = rect.get_height()
    #     ax.text(str(height), 1.05*height,
    #             '%d' % int(height),
    #             ha='center', va='bottom')


    rects1 = plt.bar(index, data[0], bar_width,
    alpha=opacity,
    color='b',
    label='TPR')

    rects2 = plt.bar(index + bar_width, data[1], bar_width,
    alpha=opacity,
    color='g',
    label='FPR')

    rects3 = plt.bar(index + 2*bar_width, data[2], bar_width,
    alpha=opacity,
    color='r',
    label='IDR')

    rects4 = plt.bar(index + 3*bar_width, data[3], bar_width,
    alpha=opacity,
    color='y',
    label='AUC')

    plt.xlabel('ML Models')
    plt.ylabel('Values')
    plt.title('MalJPEG Performance Comparison')
    plt.xticks(index + bar_width, ('Native Bayes', 'Decision Tree', 'Random Forest', 'XGBoost','LightGBM'))
    plt.legend()

    for bar in rects1:
        yval = round(bar.get_height(),4)
        plt.text(bar.get_x(), yval + .005, yval)
    
    for bar in rects2:
        yval = round(bar.get_height(),4)
        plt.text(bar.get_x(), yval + .005, yval)
    
    for bar in rects3:
        yval = round(bar.get_height(),4)
        plt.text(bar.get_x(), yval + .005, yval)
    
    for bar in rects4:
        yval = round(bar.get_height(),4)
        plt.text(bar.get_x(), yval + .005, yval )

    plt.tight_layout()
    plt.show()  

if __name__=='__main__':
    main()

    

