import os
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import datasets
import sklearn.metrics as Eval
import itertools as IT

class Pickle:
    def Saver(self, Path, Object):
       with open(Path, 'wb') as file:
           pickle.dump(Object, file)
       return True
    def Loader(self, Path):
        with open(Path, 'rb') as file:
            Object = pickle.load(file)
        return Object

def Dataset_Parameters_Loader(g):
    """
    :param g: The growing percentage of the data in float, ex. 50% should present as 50/100 = 0.5
    :return: X: The data based on the growing percentage
             Y: The ground truth labels based on the growing percentage
             uL: Classes found in ground truth labels
             k: Number of classes
             K0: Indexes of Setosa samples
             K1: Indexes of Versicolor samples
             K2: Indexes of Virginica samples
             count: The number of generated centers possibilites
    """
    # Load dataset
    datas = datasets.load_iris()
    X = datas['data']
    Y = datas['target']
    uL = np.unique(Y)
    k=len(uL)

    # Implement growing status
    Itrain = []
    for i in uL:
        I = np.argwhere(Y == i)[:, 0]
        cutoff = int(len(I) * g)
        I = I[0:cutoff]
        Itrain.extend(I)
    # Split the data
    X = X[Itrain, :]
    Y = Y[Itrain]

    Ks = []
    for i in uL:
        I = np.argwhere(Y == i)[:, 0]
        Ks.append(I)
    K0, K1, K2 = Ks

    # Count possibilites
    count = 0
    for _ in IT.product(K0, K1, K2):
        count = 1 + count

    F = []
    for r in range(2,X.shape[1]+1):
        for f in IT.combinations(np.array(range(X.shape[1])),r):
            F.append(np.array(f).tolist())
    return X, Y, K0, K1, K2, k, uL, count, F

def Method(X,Centers, Distance):
    """
    Notice: You can replace any clustering model with the code here to investigate its performance,
            for example, K-means, Density peaks, or any other clustering models depend on centers and distance
            matrix to cluster the data

    :param X: The dataset that need to be clustered
    :param Centers: The cluster centers which are represented as indexes of X
    :param Distance: The required distance matrix
    :return:
    """
    Centers = X[np.array(Centers).tolist()]
    y_pred = np.argmin(cdist(X,Centers, Distance),axis=1)
    return y_pred


################################################
"""
    You can change G values but it should be between 0 to 1 where 0 means 0% and 1 means 100%
    For DistanceMatrix, you can add more distance metrics where they should be processed by scipy.spatial.distance.cdist or your own model
    Change your required validation by changing V value 0,1,2 to choose the Validation Type from ValidationTypes list
"""
G = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
DistanceMatrix = ['braycurtis',
                  'canberra',
                  'chebyshev',
                  'cityblock',
                  'correlation',
                  'cosine',
                  'euclidean',
                  'hamming',
                  'jaccard',
                  'jensenshannon',
                  'mahalanobis',
                  'matching',
                  'minkowski',
                  'seuclidean',
                  'sqeuclidean']

V = 0

ValidationTypes = ["Distance Validation",
                   "Feature Validation",
                   "Feature and Distance Validation"]

Validation = ValidationTypes[V]

################################################
pickle_ = Pickle()
save_path = '/Results/'



if Validation == "Distance Validation":
    """
    The ResultsArray here represents the growing status impact toward distance and centers relations.
    ResultsArray for every growing status will be generated
    """
    for g_idx, g in enumerate(G):
        X, Y, K0, K1, K2, k, uL, count, F = Dataset_Parameters_Loader(g)
        ResultsArray = np.zeros([count, len(DistanceMatrix)])
        for i, centers in enumerate(IT.product(K0, K1, K2)):
            for j, distanceM in enumerate(DistanceMatrix):
                cl = Method(X,centers, distanceM)
                ResultsArray[i, j] = Eval.accuracy_score(Y, cl)
            print("Distance Validation: G=", int(g*100), "% -- C#", i, "/", count, " -- Results: \n",  np.round(ResultsArray[i, :].tolist(),4).tolist(), "\n")
        filename = "Distance_Validation_Grow_" + str(int(g*100)) + ".pkl"
        pickle_.Saver(os.path.join(save_path,filename), ResultsArray)



elif Validation == 'Feature Validation':
    """
    The ResultsArray here represents the growing status impact toward features and centers relations.
    ResultsArray for every growing status will be generated
    """
    for g_idx, g in enumerate(G):
        X, Y, K0, K1, K2, k, uL, count, F = Dataset_Parameters_Loader(g)
        ResultsArray = np.zeros([count, len(F)])
        for i, centers in enumerate(IT.product(K0, K1, K2)):
            for j, f in enumerate(F):
                cl = Method(X[:,f], centers, 'euclidean')
                ResultsArray[i, j] = Eval.accuracy_score(Y, cl)
            print("Distance Validation: G=", int(g*100), "% -- C#", i, " out of ", count, " -- Results: \n",  np.round(ResultsArray[i, :].tolist(),4).tolist(), "\n")
        filename = "Features_Validation_Grow_" + str(int(g*100)) + ".pkl"
        pickle_.Saver(os.path.join(save_path,filename), ResultsArray)


elif Validation == "Feature and Distance Validation":
    """
    The ResultsArray array here represents the features impact toward distance and centers relations. 
    ResultsArray for every Feature set in F will be generated
    """
    X, Y, K0, K1, K2, k, uL, count, F = Dataset_Parameters_Loader(1)
    for f_idx, f in enumerate(F):
        ResultsArray = np.zeros([count, len(DistanceMatrix)])
        for i, centers in enumerate(IT.product(K0, K1, K2)):
            for j, D in enumerate(DistanceMatrix):
                cl = Method(X[:, f], centers, D)
                ResultsArray[i, j] = Eval.accuracy_score(Y, cl)
            print("Feature and Distance Validation: -- C#", i, " out of ", count, " -- Results: \n",np.round(ResultsArray[i, :].tolist(), 4).tolist(), "\n")
        filename = "Features_Distance_Validation_F_" + str(f_idx) + ".pkl"
        pickle_.Saver(os.path.join(save_path,filename), ResultsArray)

        
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
