import random
import time
import numpy as np
rng = np.random.default_rng()
import math
import sys
from numpy import linalg as LA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from processing_data.BHBA_process import *


def bhba(df, y, pop, dim, lb, ub, Max_iter, alpha_bhba):
    X = initial(pop, dim, lb, ub)  # Initialize the number of honey badgers

    fitness = np.zeros([pop, 1])
    total_acc = np.zeros([pop, 4])
    for i in range(pop):
        fitness_value, class_acc = CaculateFitness2(df, y, X[i, :], alpha_bhba)
        fitness[i] = fitness_value
        total_acc[i, :] = class_acc

    # print('------------------ test on ', alpha_bhba, ' ---------------')
    fitness, sortIndex = SortFitness(fitness)  # Sort the fitness values of honey badger.
    X = SortPosition(X, sortIndex)  # Sort the honey badger.
    GbestScore = fitness[0]  # The optimal value for the current iteration.
    GbestAcc = total_acc[sortIndex[0], :]
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([Max_iter, 1])
    total_ACC = np.zeros([Max_iter, 4])

    C = 2  # constant in Eq. (3)
    beta = 6  # the ability of HB to get the food  Eq.(4)
    vec_flag = [1, -1]
    vec_flag = np.array(vec_flag)
    Xnew = np.zeros([pop, dim])
    for t in range(Max_iter):
        print("iteration: ", t)
        alpha = C * math.exp(-t / Max_iter)  # density factor in Eq. (3)
        I = Intensity(pop, GbestPositon, X)  # intensity in Eq. (2)
        Vs = random.random()
        # print("------------------- test2 ---------------------")
        for i in range(pop):
            Vs = random.random()
            F = vec_flag[math.floor((2 * random.random()))]
            for j in range(dim):
                di = GbestPositon[0, j] - X[i, j]
                if (Vs < 0.5):  # Digging phase Eq. (4)
                    r3 = np.random.random()
                    r4 = np.random.randn()
                    r5 = np.random.randn()
                    Xnew[i, j] = GbestPositon[0, j] + F * beta * I[i] * GbestPositon[0, j] + F * r3 * alpha * (
                        di) * np.abs(math.cos(2 * math.pi * r4) * (1 - math.cos(2 * math.pi * r5)))
                else:
                    r7 = random.random()
                    Xnew[i, j] = GbestPositon[0, j] + F * r7 * alpha * di  # Honey phase Eq. (6)
            # print(di)
            Xnew[i, :] = BorderCheck2(Xnew[i, :], dim)
            # introduce X_binary
            if sum(Xnew[i, :]) >= 0.1*dim:
                tempFitness, class_acc = CaculateFitness2(df, y, Xnew[i, :], alpha_bhba)
                if (tempFitness <= fitness[i]):
                    fitness[i] = tempFitness
                    X[i] = Xnew[i]
                    total_acc[i, :] = class_acc


        # for i in range(pop):
        # X[i] = BorderCheck2(X[i], dim)
        Ybest, index = SortFitness(fitness)  # Sort fitness values.
            #             print(Ybest[0])
        if (Ybest[0] <= GbestScore):
            GbestScore = Ybest[0]  # Update the global optimal solution.
            GbestPositon[0, :] = X[index[0], :]  # Sort fitness values
            GbestAcc = total_acc[index[0], :]


        total_ACC[t, :] = GbestAcc
        Curve[t] = GbestScore
        print(Curve[t], sum(GbestPositon[0]))

    return GbestScore, GbestPositon, Curve, total_ACC, GbestAcc


# # option 1: CanRNA-5 (Mendeley) data
# in_path = '../Dataset/CanRNA-5_mendeley/'
# df = pd.read_csv(in_path + 'Mendely_data.csv')
# y = df.iloc[:, -1]
# df = df.iloc[:, :-1]

# option 2: PANC-5 (UCI) data
in_path = '../Dataset/PANC-5_UCI/'
df   = pd.read_csv(in_path + 'UCI_Data.csv')
df   = df.drop(columns=["Unnamed: 0"])
df_label = pd.read_csv(in_path + 'UCI_Label.csv')
label = LabelEncoder()
y = label.fit_transform(df_label['Class'])

# option 3: GSE211692 (GSE) data
# in_path = '../Dataset/GSE211692/'
# df = pd.read_csv(in_path + 'GSE211692_dataset.csv', header=None)
# y = pd.read_csv(in_path + 'labels.csv')
# label = LabelEncoder()
# y = label.fit_transform(y['ID_REF'])

rng = np.random.default_rng()
pop1 = [20, 40, 50, 100, 150]
pop = 50                   # Honey Badger population size.
MaxIter = 10               # Maximum number of iterations.
dim = len(df.columns)                 # The dimension.
fl=-2                    # The lower bound of the search interval.
ul=2                      # The upper bound of the search interval.
# alpha_table = [0.80, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
alpha_table = np.arange(0.8, 1.001, 0.05)
alpha_table = np.round(alpha_table, 2)
lb = fl*np.ones([dim, 1])
ub = ul*np.ones([dim, 1])
# print(lb)
results = np.zeros([len(alpha_table), 7])
results2 = []
for i, alpha_bhba in enumerate(alpha_table):
# for i, pop in enumerate(pop1):
    print('------------------ test on ', alpha_bhba, ' ---------------')
    time_start = time.time()
    GbestScore, GbestPositon, Curve, total_ACC, GbestAcc = bhba(df, y, pop, dim, lb, ub, MaxIter, 0.8)
    time_end = time.time()
    print(f"The running time is: {time_end  - time_start } s")
    print('The optimal value：',GbestScore)
    # print('The optimal solution：',GbestPositon)
    print('Number of selected feature is: ', sum(GbestPositon[0]))
    results[i, 0] = sum(GbestPositon[0])
    results[i, 1] = GbestScore[0]
    results[i, 2] = time_end - time_start
    results[i, 3] = GbestAcc[0, 0]            #class accuracy
    results[i, 4] = GbestAcc[0, 1]            #F1 measure
    results[i, 5] = GbestAcc[0, 2]            #Recall
    results[i, 6] = GbestAcc[0, 3]            #Precision
    results2.append(Curve)
    results2.append(total_ACC)
    results2.append(GbestPositon[0])
    rng = np.random.default_rng()

    suffix_name = '_SVM_0.8.npy' # _pop_' + str(pop) +
    np.save('tmp/results' + suffix_name, results)
    np.save('tmp/curve' + suffix_name, [results2[j] for j in range(0, 3*i + 1, 3)])    #   (1, 2*(i+1), 2)
    np.save('tmp/acc' + suffix_name, [results2[j] for j in range(1, 3*i + 2, 3)])  #   (0, 2*(i+1) - 1, 2)
    np.save('tmp/GbestPosition' + suffix_name, [results2[j] for j in range(2, 3*i + 3, 3)])
    end_of_loop = 0

suffix_name = '_NB.npy'
np.save('results' + suffix_name, results)
np.save('curve' + suffix_name, [results2[j] for j in range(0, 3*i + 1, 3)])        # 42
np.save('acc' + suffix_name, [results2[j] for j in range(1, 3*i + 2, 3)])    # 41
np.save('GbestPosition' + suffix_name, [results2[j] for j in range(2, 3*i + 3, 3)])

print('---------------------------------------- End of Test ----------------------------------------')