import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score


#---------------Feature Extraction-----------------------------------
def featureExtraction(data, num):
    extract_data = pd.DataFrame(np.random.rand(num * 13).reshape(num, 13), columns=['number', 'x_mean', 
                                'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_max', 'y_max', 'z_max', 
                                'x_min', 'y_min', 'z_min'])
    # extract_data = extract_data.applymap("${0:.2f}".format)
    for i in range(len(data)):
        r = data.iloc[i,3:39]
        num = r.size - r.isnull().sum()
        num =  num/3
        extract_data['number'][i] = num

        x_array = np.array([r['X0'], r['X1'], r['X2'], r['X3'], r['X4'], r['X5'], r['X6'], r['X7'], r['X8'], 
                            r['X9'], r['X10'], r['X11']])
        y_array = np.array([r['Y0'], r['Y1'], r['Y2'], r['Y3'], r['Y4'], r['Y5'], r['Y6'], r['Y7'], r['Y8'], 
                            r['Y9'], r['Y10'], r['Y11']])
        z_array = np.array([r['Z0'], r['Z1'], r['Z2'], r['Z3'], r['Z4'], r['Z5'], r['Z6'], r['Z7'], r['Z8'], 
        r['Z9'], r['Z10'], r['Z11']])

        extract_data['x_mean'][i] = np.nanmean(x_array)
        extract_data['y_mean'][i] = np.nanmean(y_array)
        extract_data['z_mean'][i] = np.nanmean(z_array)
        extract_data['x_std'][i] = np.nanstd(x_array, ddof = 1)
        extract_data['y_std'][i] = np.nanstd(y_array, ddof = 1)
        extract_data['z_std'][i] = np.nanstd(z_array, ddof = 1)
        extract_data['x_max'][i] = np.nanmax(x_array)
        extract_data['y_max'][i] = np.nanmax(y_array)
        extract_data['z_max'][i] = np.nanmax(z_array)
        extract_data['x_min'][i] = np.nanmin(x_array)
        extract_data['y_min'][i] = np.nanmin(y_array)
        extract_data['z_min'][i] = np.nanmin(z_array)

    extract_data.insert(0,'User',data['User'])
    extract_data.insert(0,'Class',data['Class'])
    
    return extract_data

#---------------Standardization-----------------------------------
def standardScaler(train_data, test_data, method):
    if method == "Standardization":
        #-------standard-------
        scaler = StandardScaler()
    else:
        #-------normalized-------
        scaler = Normalizer()
    train_data_new = train_data.drop('User',axis=1)
    train_data_new = train_data_new.drop('Class',axis=1)
    scaler.fit(train_data_new)
    training_data_Standard = pd.DataFrame(scaler.transform(train_data_new),columns=['number', 'x_mean', 
                                'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_max', 'y_max', 'z_max', 
                                'x_min', 'y_min', 'z_min'])
    training_data_Standard.insert(0,'User', train_data['User'])
    training_data_Standard.insert(0,'Class', train_data['Class'])
    test_data_new = test_data.drop('User',axis=1)
    test_data_new = test_data_new.drop('Class',axis=1)
    testing_data_Standard = pd.DataFrame(scaler.transform(test_data_new),columns=['number', 'x_mean', 
                                'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_max', 'y_max', 'z_max', 
                                'x_min', 'y_min', 'z_min'])
    testing_data_Standard.insert(0,'User', test_data['User'])
    testing_data_Standard.insert(0,'Class', test_data['Class'])
    return training_data_Standard, testing_data_Standard



#---------------Feature Selection-----------------------------------
def featureSelection(train_data, test_data, method):
    train_data_new = train_data.drop('User',axis=1)
    train_data_new = train_data_new.drop('Class',axis=1)
    train_label = train_data['Class']
    label_np = train_label.to_numpy()
    label_np = np.ravel(label_np)
    if method == "PCA":
        select = PCA(n_components=0.99)
        select.fit(train_data_new)
        pass
    else:
        select = SelectKBest()
        select.fit(train_data_new, label_np)
        pass
    training_data_pca = pd.DataFrame(select.transform(train_data_new))
    training_data_pca.insert(0,'User', train_data['User'])
    training_data_pca.insert(0,'Class', train_data['Class'])
    test_data_new = test_data.drop('User',axis=1)
    test_data_new = test_data_new.drop('Class',axis=1)
    testing_data_pca = pd.DataFrame(select.transform(test_data_new))
    testing_data_pca.insert(0,'User', test_data['User'])
    testing_data_pca.insert(0,'Class', test_data['Class'])
    return training_data_pca, testing_data_pca


def get_X_and_label(data):
    label = data['Class']
    label_np = label.to_numpy()
    label_np = np.ravel(label_np)
    X = data.iloc[:,2:]
    X_np = X.to_numpy()
    return X_np, label_np

#---------------model for default input-----------------------------------
def defaultModel(train_data, test_data,model):
    acc_test_avg = 0
    acc_avg = 0
    acc_train_avg = 0
    f_avg = 0
    cm_final = np.zeros(25)
    for i in range(12):
        if i!=3 and i!=4 and i!=7:
            data_valid = train_data[train_data['User'] == i]
            data_train = train_data[train_data['User'] != i]
            X_tr,l_tr = get_X_and_label(data_train)
            X_v,l_v = get_X_and_label(data_valid)
            X_te,l_te = get_X_and_label(test_data)
            if model == "NaiveBayes":
                clf = GaussianNB()
            elif model == "Perceptron":
                clf = Perceptron(warm_start = True)
            elif model == "knn":
                clf = KNeighborsClassifier()
            elif model == "svm":
                clf = SVC()
            elif model == "MLF":
                clf =MLPClassifier()
            clf.fit(X_tr, l_tr)
            l_te_pred = clf.predict(X_te)
            acc_v = clf.score(X_v, l_v)
            acc_avg += acc_v
            acc_te = clf.score(X_te,l_te)
            acc_test_avg += acc_te
            acc_tr = clf.score(X_tr,l_tr)
            acc_train_avg += acc_tr
            cm = confusion_matrix(l_te,l_te_pred).ravel()
            cm_final += cm
            f = f1_score(l_te, l_te_pred, average=None)
            f_avg += f
    acc_test_avg = acc_test_avg/9
    acc_avg = acc_avg/9
    acc_train_avg = acc_train_avg/9 
    cm_final = cm_final/9
    f_avg = f_avg/9
    print("Its avg score is:", acc_avg)
    print("Its avg test score is:", acc_test_avg)
    print("Its avg train score is:", acc_train_avg)
    cm_final = pd.DataFrame(cm_final.reshape((5,5)), columns=['Predict1', 'Predict2', 'Predict3', 'Predict4','Predict5'], 
                            index=['Actual1', 'Actual2', 'Actual3', 'Actual4','Actual5'])
    print(cm_final)
    print("Its avg f1 score is:", f_avg)


#---------------Perceptron with 50 iteration-----------------------------------
def pp(train_data, test_data):
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
                neigh = Perceptron(random_state=np.random, warm_start = True)
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
    best_acc_va = np.amax(knn)
    best_acc_train = np.amax(knn_train)
    print("Best perceptron test score is: ",best_acc_test )
    print("Best perceptron validation score is: ",best_acc_va)
    print("Best perceptron train score is: ",best_acc_train)


