import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score

def get_X_and_label(data):
    label = data['Class']
    label_np = label.to_numpy()
    label_np = np.ravel(label_np)
    X = data.iloc[:,2:]
    X_np = X.to_numpy()
    return X_np, label_np


#---------------SVM-----------------------------------
#Find the best performance of each svm to get each best c value
def detailedSVM(train_data, test_data,method):
    c = np.logspace(-1.5,1.5,30)
    ACC = np.zeros(30,dtype=float)
    ACC_test = np.zeros(30,dtype=float)
    for i in range(len(c)):
        for x in range(12):
            if x!=3 and x!=4 and x!=7:
                data_valid = train_data[train_data['User'] == x]
                data_train = train_data[train_data['User'] != x]
                X_tr,l_tr = get_X_and_label(data_train)
                X_v,l_v = get_X_and_label(data_valid)
                X_te,l_te = get_X_and_label(test_data)
                if method == 'sigmoid':
                    clf = SVC(C=c[i],kernel='sigmoid',gamma='auto')
                elif method == 'linear':
                    clf = SVC(C=c[i],kernel='linear')
                elif method == 'rbf':
                    clf = SVC(C=c[i],kernel='rbf',gamma='auto')
                elif method == 'poly':
                    clf = SVC(C=c[i],kernel='poly',gamma='auto')
                clf.fit(X_tr, l_tr)
                acc_v = clf.score(X_v, l_v)
                acc_te = clf.score(X_te,l_te)
                ACC[i] += acc_v
                ACC_test[i] += acc_te
        ACC[i] = (ACC[i])/9
        ACC_test[i] = (ACC_test[i])/9
        print(i)
    best_acc_test = np.amax(ACC_test)
    best_acc_test_ind = np.argmax(ACC_test)
    best_c_test = c[best_acc_test_ind]
    best_acc = np.amax(ACC)
    best_acc_ind = np.argmax(ACC)
    best_c = c[best_acc_ind]
    print("Best acc of ", method, " SVM in test is ", best_acc_test, "whith C: ",best_c_test)
    print("Best acc of ", method, " SVM in validation is ", best_acc, "whith C: ",best_c)
    return best_c


#---------------SVM Confuson matrix and f1 score -----------------------------------
#Using best score in vlidation to get its c and get the test data's confuson matrix and f1 score
def CCsvm(train_data, test_data,method,best_c):
    acc_test_avg = 0
    acc_avg = 0
    acc_train_avg = 0
    f_avg = 0
    cm_avg = np.zeros(25)
    for i in range(12):
        if i!=3 and i!=4 and i!=7:
            data_valid = train_data[train_data['User'] == i]
            data_train = train_data[train_data['User'] != i]
            X_tr,l_tr = get_X_and_label(data_train)
            X_v,l_v = get_X_and_label(data_valid)
            X_te,l_te = get_X_and_label(test_data)
            if method == 'sigmoid':
                clf = SVC(C=best_c,kernel='sigmoid',gamma='auto')
            elif method == 'linear':
                clf = SVC(C=best_c,kernel='linear')
            elif method == 'rbf':
                clf = SVC(C=best_c,kernel='rbf',gamma='auto')
            elif method == 'poly':
                clf = SVC(C=best_c,kernel='poly',gamma='auto')
            clf.fit(X_tr, l_tr)
            l_te_pred = clf.predict(X_te)
            acc_v = clf.score(X_v, l_v)
            acc_avg += acc_v
            acc_te = clf.score(X_te,l_te)
            acc_test_avg += acc_te
            acc_tr = clf.score(X_tr,l_tr)
            acc_train_avg += acc_tr
            cm = confusion_matrix(l_te,l_te_pred).ravel()
            cm_avg += cm
            f = f1_score(l_te, l_te_pred, average=None)
            f_avg += f
    acc_test_avg = acc_test_avg/9
    acc_avg = acc_avg/9
    acc_train_avg = acc_train_avg/9 
    cm_avg = cm_avg/9
    f_avg = f_avg/9
    print("Its avg score is:", acc_avg)
    print("Its avg test score is:", acc_test_avg)
    print("Its avg train score is:", acc_train_avg)
    cm_avg = pd.DataFrame(cm_avg.reshape((5,5)), columns=['Predict1', 'Predict2', 'Predict3', 'Predict4','Predict5'], 
                            index=['Actual1', 'Actual2', 'Actual3', 'Actual4','Actual5'])
    print(cm_avg)
    print("Its avg f1 score is:", f_avg)