outFile = 'tempfile.out'
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment

'''
Author: Balaji (1001576836)
'''
def k_means(clusters, nparray):
    # n_clusters : int, optional. The number of clusters to form as well as the number of centroids to generate.
    # random_state : int (0-42) Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See Glossary.

    k = KMeans(n_clusters=clusters).fit(nparray[1:, ].T)
    confusionMatrix = confusion_matrix(nparray[0], k.labels_ + 1)
    print(confusionMatrix)
    cluster_acc(nparray[0], k.labels_ + 1)

#code given in class
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    reorder = w[1:, ind[1:, 1]]
    print("Reorder")
    print(reorder)

    print("Accuracy: ", (sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size) * 100)

#code from project1
def pickDataClass(filename, class_ids):
    nd_data1 = np.genfromtxt(filename, delimiter=',')
    classids_col = []
    for i in class_ids:
        a = np.where(nd_data1[0] == i)
        classids_col.extend(np.array(a).tolist())
    classids_col = [j for k in classids_col for j in k]
    np.savetxt(outFile, nd_data1[:, classids_col], fmt="%i", delimiter=',')


print("Project 2")
print("Tasks to Complete")
print("************************TASK A******************************************")
print("Picking 100 AT and T images. Open tempfile.out to see the output of this step")
pickDataClass("ATNTFaceImages400.txt", range(1, 11))
nparray = np.genfromtxt(outFile, delimiter=",")
k_means(10, nparray)
print("************************TASK B******************************************")
print("Picking 400 AT and T images. Open tempfile.out to see the output of this step")
pickDataClass("ATNTFaceImages400.txt", range(1, 41))
nparray = np.genfromtxt(outFile, delimiter=",")
k_means(40, nparray)
print("************************TASK C ******************************************")
print("Running K-Means on Hand Written Data Set for n=26")
nparray = np.genfromtxt("HandWrittenLetters.txt", delimiter=",")
k_means(26, nparray)
