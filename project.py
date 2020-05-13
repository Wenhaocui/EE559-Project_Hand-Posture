import pandas as pd
import numpy as np
import util
import svm_detailed as sd
import knn
from sklearn.preprocessing import StandardScaler

training_data = pd.read_csv('D_train.csv',delimiter =',')
testing_data = pd.read_csv('D_test.csv',delimiter =',')
# displays all columns
pd.set_option('display.max_columns', None)

#Data extraction and clean the null in training and testing set
train_data_extracted = util.featureExtraction(training_data, 13500)
test_data_extracted = util.featureExtraction(testing_data, 21099)

training_data_Standard, testing_data_Standard = util.standardScaler(train_data_extracted, test_data_extracted, "Standardization")
training_data_normal, testing_data_normal = util.standardScaler(train_data_extracted, test_data_extracted, "Normalizer")

#Compare the standardization and normalization for scaling
util.defaultModel(training_data_Standard, testing_data_Standard,"NaiveBayes")
util.defaultModel(training_data_normal, testing_data_normal,"NaiveBayes")
util.defaultModel(training_data_Standard, training_data_Standard,"Perceptron")
util.defaultModel(training_data_normal, testing_data_normal,"Perceptron")

training_data_pca, testing_data_pca = util.featureSelection(training_data_Standard, testing_data_Standard, "PCA")
training_data_kb, testing_data_kb = util.featureSelection(training_data_Standard, testing_data_Standard, "KB")

#Compare the PCA and k Best for dimensionality adjustment
util.defaultModel(training_data_pca, testing_data_pca,"NaiveBayes")
util.defaultModel(training_data_kb, testing_data_kb,"NaiveBayes")

#Compare the default perceptron
util.defaultModel(training_data_pca, testing_data_pca,"Perceptron")
#Shuffling the training set to get the best perceptron performance in 50 times
util.pp(training_data_pca, testing_data_pca)

#Compare the default NaiveBayes（baseline)
util.defaultModel(training_data_pca, testing_data_pca,"NaiveBayes")

#Compare the default MLF
util.defaultModel(training_data_pca, testing_data_pca,"MLF")

#Get the best c and kernel in SVM from validation score and get their test score and confusion matrix
best_rbf_c = sd.detailedSVM(training_data_pca, testing_data_pca, 'rbf')
sd.CCsvm(training_data_pca, testing_data_pca, 'rbf',best_rbf_c)

best_poly_c = sd.detailedSVM(training_data_pca, testing_data_pca, 'poly')
sd.CCsvm(training_data_pca, testing_data_pca, 'poly',best_poly_c)

best_linear_c = sd.detailedSVM(training_data_pca, testing_data_pca, 'linear')
sd.CCsvm(training_data_pca, testing_data_pca, 'linear',best_linear_c)

best_sigmoid_c = sd.detailedSVM(training_data_pca, testing_data_pca, 'sigmoid')
sd.CCsvm(training_data_pca, testing_data_pca, 'sigmoid',best_sigmoid_c)


#Get the best k in KNN from validation score and get their test score and confusion matrix
best_k = knn.KNN(training_data_pca, testing_data_pca)
knn.oneKNN(training_data_pca, testing_data_pca, best_k)

#---------------randomclassifier-----------------------------------
def randomclassifier():
    rc = np.random.randint(1,5,21099)
    label = testing_data_pca['Class']
    label_np = label.to_numpy()
    t = np.sum(rc == label_np)
    acc = t/21099
    print(acc)

#baseline 2
randomclassifier()