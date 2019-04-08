import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from csv import reader
import pandas as pd 
from sklearn.feature_selection import SelectKBest, f_classif
'''
Output of pickdataclass & Input splitData2testtrain is tempfile.out
'''
outFile = 'tempfile.out'
SVM = svm.LinearSVC()
"""

import numpy as np
from csv import reader
import pandas as pd 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import LinearSVC

with open('GenomeTrainXY.txt') as file:
    csv_reader = reader(file)
    for line in csv_reader:
        vals = [line.split(',') for line in file]
        for i in range(len(vals)):
            vals[i][-1] = vals[i][-1].replace('\n','')
arr = np.asarray(vals)
ar = arr.transpose()
class_labels = np.asarray([x for x in line])
features = np.delete(ar, (0), axis=0)

F = SelectKBest(f_classif,k=100).fit_transform(ar,class_labels)
val = SelectKBest(f_classif,k=100).fit(ar,class_labels).get_support()
df = pd.DataFrame(F)
df.to_csv('a.csv', index=False, header=False)
output=open('xyz.txt','w')

with open('a.csv', encoding='ascii') as f:
    for row in f:
        output.write(row)
top_features = np.asarray(F)
clf = LinearSVC()
clf.fit(top_features, class_labels)
with open('GenomeTestX.txt') as file:
    csv_reader = reader(file)
    for line in csv_reader:
        vals = [line.split(',') for line in file]
        for i in range(len(vals)):
            vals[i][-1] = vals[i][-1].replace('\n','')
arr1 = np.asarray(vals)
ar1 = arr.transpose()
for x,y in zip(val, class_labels):
    if x:
        
"""
#Classification of data using KNN, Linear regression, centroid and SVM method

def Ftest():
    with open('GenomeTrainXY.txt') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            vals = [line.split(',') for line in file]
            for i in range(len(vals)):
                vals[i][-1] = vals[i][-1].replace('\n','')
    arr = np.asarray(vals)
    ar = arr.transpose()
    class_labels = np.asarray([x for x in line])
    features = np.delete(ar, (0), axis=0)
    F = SelectKBest(f_classif,k=100).fit_transform(ar,class_labels)
    df = pd.DataFrame(F)
    df.to_csv('a.csv', index=False, header=False)
    output=open('xyz.txt','w')
    with open('a.csv', encoding='ascii') as f:
        for row in f:
            output.write(row)
    
def taskB():
    '''
    data set: ATNT-face-image400.txt  :
        Text file. 
            1st row is cluster labels. 
            2nd-end rows: each column is a feature vectors (vector length=28x23).
    Total 40 classes. each class has 10 images. Total 40*10=400 images
    '''
    Ftest()
    SVM_scores = []
    centroid_scores = []
    KNN_scores = []
    linear_scores = []
    print('Computing' + '.' * 25)
    #considering the data in steps of 2 i.e. 0, 2, 4, 6, 8
    for i in range(0, 10, 2):
        testVector, testLabel, trainVector, trainLabel = splitData2TestTrain('xyz.txt', 10, str(i) + ':' + str(i+1)) # taking adjacent data for test
        linear_scores.append(linear(trainVector, testVector, trainLabel, testLabel))
        svmMatrix = svmClassifier(trainVector.transpose(), trainLabel, testVector.transpose(), testLabel)
        #Finalsvm stores the accuracy score of SVM model
        finalsvm = (SVM.score(svmMatrix, testLabel))*100
        SVM_scores.append(finalsvm)
        res_centroid = centroid(trainVector, trainLabel, testVector, testLabel)
        #Finding  the accuracy comparision from testLabel and res_centroid
        test_error3 = testLabel - res_centroid
        acc_padding3 = (1 - np.nonzero(test_error3)[0].size / float(len(test_error3))) * 100
        centroid_scores.append(acc_padding3)
        finalknn = kNearestNeighbor(trainVector, trainLabel, testVector, 5)
        #Finding  the accuracy comparision from testLabel and finalknn
        test_error4 = testLabel - finalknn
        acc_padding4 = (1 - np.nonzero(test_error4)[0].size / float(len(test_error4))) * 100
        KNN_scores.append(acc_padding4)
    print('\nThe average accuracies are...')
    print('\nSVM : %f' % (sum(SVM_scores) / len(SVM_scores)))
    print(SVM_scores)
    print('\nCentroid : %f' % (sum(centroid_scores) / len(centroid_scores)))
    print(centroid_scores)
    print('\nKNN (K = 5) : %f' % (sum(KNN_scores) / len(KNN_scores)))
    print(KNN_scores)
    print('\nLinear Regression : %f' % (sum(linear_scores) / len(linear_scores)))
    print(linear_scores)
    '''with open('ipop.txt','w') as f:
        f.write('\n')
        f.write('Output: \n')
        f.write('Average Accuracy for Task B:\n')
        f.write('SVM : ')
        f.write(str((sum(SVM_scores) / len(SVM_scores))))
        f.write('\nCentroid : ')
        f.write(str( (sum(centroid_scores) / len(centroid_scores))))
        f.write('\nLinear : ')
        f.write(str((sum(KNN_scores) / len(KNN_scores))))
        f.write('\nKNN : ')
        f.write(str((sum(linear_scores) / len(linear_scores))))'''
    
'''
 splitData2TestTrain takes filename, number_per_class, test_instances
 split the data into testVector, testLabel, trainVector, trainLabel
 Get list of train instances, test instances, strip them and add into respective matrix.
'''
#Splitting the data into train data, train label, test data and test label based on the number of data points per class and start and end instances of test labels
def splitData2TestTrain(filename, number_per_class, test_instances):
    first_classid, last_classid = test_instances.split(":")
    nd_data2 = np.genfromtxt(filename, delimiter=',')
    train_data = []
    test_data = []
    #from the given test instances, test labels are stored in list and the remaining instances are stored in train labels list
    test_classids = list(range(int(first_classid), int(last_classid)+1))
    train_classids = list((set(list(range(0, number_per_class))) - set(test_classids)))
    for i in range(0, nd_data2[0].size, number_per_class):
        train_list = [x + i for x in train_classids]
        train_list.sort()
        test_list = [x + i for x in test_classids]
        test_list.sort()
        if len(train_data) == 0:
            train_data = nd_data2[:, train_list]
        else:
            train_data = np.concatenate((train_data, nd_data2[:, train_list]), axis=1)
        if len(test_data) == 0:
            test_data = nd_data2[:, test_list]
        else:
            test_data = np.concatenate((test_data, nd_data2[:, test_list]), axis=1)
    #train_data[0] and test_data[0] contains all the train labels and test labels while train_data[1:,] and test_data[1:,0] contains all the train data and test data respectively
    return test_data[1:, ], test_data[0], train_data[1:, ], train_data[0]

#Training data is fitted into SVM model with scikit library and the the values are predicted with the test data
def svmClassifier(train_data, trainLabel, test_data, testLabel):
    SVM.fit(train_data, trainLabel)
    SVM.predict(test_data)
    return test_data

'''

 centroid method compares the eucledean distance between the 
 nearest centroid. 

'''
#Centroid method classification is based on the euclidean distance the data point and the nearest centroid 
def centroid(trainVector, trainLabel, testVector, testLabel):
    result = []
    mean_list = []
    for j in range(0, len(trainVector[0]), 8):
        colavg = [trainLabel[j]]
        for i in range(len(trainVector)):
            colavg.append(np.mean(trainVector[i, j:j + 7]))
        if not len(mean_list):
            mean_list = np.vstack(colavg)
        else:
            mean_list = np.hstack((mean_list, (np.vstack(colavg))))
    for l in range(len(testVector[0])):
        linear_dist = []
        for n in range(len(mean_list[0])):
            euclid_dist = np.sqrt(np.sum(np.square(testVector[:, l] - mean_list[1:, n])))
            linear_dist.append([euclid_dist, int(mean_list[0, n])])
            linear_dist = sorted(linear_dist, key=lambda linear_dist: linear_dist[0])
        result.append(linear_dist[0][1])
    return result

'''
 Linear regression:
    Xtest_padding is formed by adding ones to bottom of Xtest
    Xtrain_padding is formed by adding ones to bottom of Xtrain
    Ytrain_Indent forms array with class label index as 1 other are zero which returns Accuracy. 
'''
#The following is a Python code for Linear Regression
def linear(Xtrain, Xtest, Ytrain, Ytest):
    counter = 0
    N_train=len(Xtrain[0])
    N_test=len(Xtest[0])
    A_train = np.ones((1, N_train))
    A_test = np.ones((1, N_test))
    Xtrain_padding = np.row_stack((Xtrain, A_train))
    Xtest_padding = np.row_stack((Xtest, A_test))
    element, index, count = np.unique(Ytrain, return_counts=True, return_index=True) #Ytrain : indicator matrix
    element = Ytrain[np.sort(index)]
    Ytrain_Indent = np.zeros((int(max(element)), count[0] * len(element)))
    for i, j in zip(count, element):
        Ytrain_Indent[int(j) - 1, counter * i:counter * i + i] = np.ones(i)
        counter += 1
    #computing regression coefficients
    B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain_Indent.T) # (XX')^{-1} X  * Y'  
    Ytest_padding = np.dot(B_padding.T, Xtest_padding)
    Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1
    test_error = Ytest - Ytest_padding_argmax
    acc_padding = (1 - np.nonzero(test_error)[0].size / float(len(test_error))) * 100
    return acc_padding

'''
 This function implements kNearestNeighbor using euclidean distance which finds
 the least k out of dominant k
'''
def kNearestNeighbor(Xtrain, Ytrain, Xtest, k):
    N_test = len(Xtest[0])
    N_train = len(Xtrain[0])
    result = []
    for i in range(N_test):
        points = []
        distances_array = []
        test_curr = Xtest[:, i]
        for j in range(N_train):
            dist = np.sqrt(np.sum(np.square(test_curr - Xtrain[:, j])))
            distances_array.append([dist, j])
            distances_array = sorted(distances_array)
        for j in range(k):
            index = distances_array[j][1]
            points.append(Ytrain[index])
        result.append(max(set(points), key=points.count))

    result = list(int(i) for i in result)
    return result

print('Enter the task:\n')
argv = str(input())

if argv.upper() == 'B':
    # Task B
    print('''TASK B : \n 
                On ATNT data, running 5-fold cross-validation (CV) using  each of the
                four classifiers: KNN, centroid, Linear Regression and SVM and presenting all 5 accuracy numbers
                for each classifier and the average of these 5 accuracy numbers.''')
    taskB()

	
	
	
