import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

def get_X_and_label(data):
    label = data['Class']
    label_np = label.to_numpy()
    label_np = np.ravel(label_np)
    X = data.iloc[:,2:]
    X_np = X.to_numpy()
    return X_np, label_np

#---------------KNN-----------------------------------
#Find the best performance of each KNN to get each best n_neighbors value
def KNN(train_data, test_data):
    knn = np.zeros(50,dtype=float)
    knn_test = np.zeros(50,dtype=float)
    knn_train = np.zeros(50,dtype=float)
    for j in range(50):
        for i in range(12):
            if i!=3 and i!=4 and i!=7:
                data_valid = train_data[train_data['User'] == i]
                data_train = train_data[train_data['User'] != i]
                X_tr,l_tr = get_X_and_label(data_train)
                X_v,l_v = get_X_and_label(data_valid)
                X_te,l_te = get_X_and_label(test_data)
                neigh = KNeighborsClassifier(n_neighbors=(j+1))
                neigh.fit(X_tr, l_tr)
                acc = neigh.score(X_v, l_v)
                knn[j] += acc
                acc_te = neigh.score(X_te,l_te)
                knn_test[j] += acc_te
                acc_tr = neigh.score(X_tr,l_tr)
                knn_train[j] += acc_tr
        print(j)
        knn[j] = knn[j]/9
        knn_test[j] = knn_test[j]/9
        knn_train[j] = knn_train[j]/9
    best_acc_test = np.amax(knn_test)
    best_acc_test_ind = np.argmax(knn_test)
    best_acc_va = np.amax(knn)
    best_acc_ind = np.argmax(knn)
    best_acc_train = np.amax(knn_train)
    best_acc_train_ind = np.argmax(knn_train)
    print("Best knn test score is: ",best_acc_test, "with", (best_acc_test_ind+1), " parameters.")
    print("Best knn validation score is: ",best_acc_va, "with", (best_acc_ind+1), " parameters.")
    print("Best knn train score is: ",best_acc_train, "with", (best_acc_train_ind+1), " parameters.")
    return (best_acc_ind+1)



#---------------KNN-----------------------------------
def oneKNN(train_data, test_data, best_n):
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
            neigh = KNeighborsClassifier(n_neighbors=best_n)
            neigh.fit(X_tr, l_tr)
            l_te_pred = neigh.predict(X_te)
            acc_v = neigh.score(X_v, l_v)
            acc_avg += acc_v
            acc_te = neigh.score(X_te,l_te)
            acc_test_avg += acc_te
            acc_tr = neigh.score(X_tr,l_tr)
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


