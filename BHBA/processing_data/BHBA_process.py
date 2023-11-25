# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix, make_scorer
# from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


rng = np.random.default_rng()
import math
import sys
from numpy import linalg as LA
import warnings
warnings.filterwarnings("error")
def process_data(df, y, position):
    df = drop_columns(df, position)
    
    # Convert the M to 1 and B to 0
    label = LabelEncoder()
    y = label.fit_transform(y)

    # Spilt the train and test data
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

    # we used 30% test data

    try:
        NB_model = GaussianNB().fit(X_train, y_train)
        y_pred = NB_model.predict(X_test)  # NB_predictions

    except:
        print('eee')


    # y_pred_train = dtree_model.predict(X_train)
    try:
        class_acc = classification_accuracy(y_test, y_pred)
    except:
        class_acc = [0, 0, 0, 0]
    
      
    return class_acc

def classification_accuracy(y_actual, y_hat):
        recall = recall_score(y_actual, y_hat, average='weighted', labels=np.unique(y_hat))
        precision = precision_score(y_actual, y_hat, average='weighted', labels=np.unique(y_hat))
        f1_score_weighted = f1_score(y_actual, y_hat, average='weighted', labels=np.unique(y_hat))
        class_acc = accuracy_score(y_actual, y_hat)

        return (class_acc, f1_score_weighted, recall, precision)
def drop_columns(df, position):
    # print(position)
    col_drop = []
    for i, index in enumerate(position):
        if position[i] == 0: 
          col_drop.append(i) 

    df_1 = df.drop(df.columns[col_drop], axis=1)

    return df_1

def Binarization_fun(position):
    for i in range(0, len(position)):
      position[i] = 1 if position[i] > 0.5 else 0

    return position

def classification_accuracy_old(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    
    class_acc = float((TP+TN)) / float((TP+FP+TN+FN))
    
    if TP == 0 and FN == 0 :
        recall = 0
    else:
        recall  = float(TP) / float(TP + FN)
    
    if TP == 0 and FP == 0:
        precision = 0
    else:
        precision = float(TP) / float( TP + FP )         
    
    return (class_acc, recall, precision)

# This function is to initialize the Honey Badger population.
def initial(pop, dim, ub, lb):
    # X is a pop*dim vector
    # position is each row of X
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
        
        X[i, :] = Binarization_fun(X[i, :])
        
    return X

# Calculate fitness values for each Honey Badger.
def CaculateFitness1(df, y, position):
    fitness = process_data(df, y, position)
    return fitness

def CaculateFitness2(df, y, position, alpha_bhba):
    class_acc = process_data(df, y, position)
    tmp_cp = class_acc[0]
    # fitness = alpha_bhba * (1 - tmp_cp)*0.95 + (1 - alpha_bhba) * (1 - sum(position)/len(position))*0.05
    fitness = alpha_bhba * (1 - tmp_cp) + (1 - alpha_bhba) * (1 - sum(position)/len(position))

    return fitness, class_acc
# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)#[::-1]
    index = np.argsort(Fit, axis=0)#[::-1]
    return fitness,index


# Sort the position of the Honey Badger according to fitness.
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


# Boundary detection function.
def BorderCheck1(X,lb,ub,dim):
        for j in range(dim):
            if X[j]<lb[j]:
                X[j] = lb[j]
            elif X[j]>ub[j]:
                X[j] = ub[j]
        return X


def BorderCheck2(X, dim):
    X1 = np.tanh(X)
    for j in range(dim):
        if (X1[j]) > 0.25:
            X1[j] = 1
        else:
            X1[j] = 0
    return X1


def Intensity(pop, GbestPositon, X):
    epsilon = 0.00000000000000022204
    di = np.zeros(pop)
    S = np.zeros(pop)
    I = np.zeros(pop)
    for j in range(pop):
        if (j < pop - 1):
            di[j] = LA.norm([[X[j, :] - GbestPositon + epsilon]])
            S[j] = LA.norm([X[j, :] - X[j + 1, :] + epsilon])
            di[j] = np.power(di[j], 2)
            S[j] = np.power(S[j], 2)
        else:
            di[j] = LA.norm([[X[-1, :] - GbestPositon + epsilon]])
            S[j] = LA.norm([[X[-1, :] - X[1, :] + epsilon]])
            di[j] = np.power(di[j], 2)
            S[j] = np.power(S[j], 2)

    for i in range(pop):
        n = random.random()
        I[i] = n * S[i] / (4 * math.pi * di[i] + epsilon)

    return I