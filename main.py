import random

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import cm

colours = ["b", "g", "r", "c", "m", "y", "k", "w"]
colors = ['blue', 'red', 'green', 'yellow', 'purple', 'darkblue', 'pink', 'orange']

MAX_ITERATIONS = 10
Js = []

def getJ(df, centroidsX, centroidsY):

    J = 0

    for i in range(len(list(df['x']))):
        xval = df['x'][i]
        yval = df['y'][i]
        cluster = df['c'][i]

        cx = centroidsX[cluster]
        cy = centroidsY[cluster]

        J = J + getDistance(cx, cy, xval, yval)

    return J

def getDistance(x1, y1, x2, y2):
    return (x1-x2)**2 + (y1-y2)**2

def getRandomCentroids(df, k):
    x = df['x']
    y = df['y']
    selectedX = random.sample(list(x), k)
    selectedY = random.sample(list(y), k)

    """for i in range(k):
        print(selectedX[i] in list(x))
        print(selectedY[i] in list(y))"""

    return selectedX, selectedY

def generateData(k, size):
    x = []
    y = []

    for cluster in range(k):
        meanX = np.random.uniform(0, 70)
        meanY = np.random.uniform(0, 70)
        sigma = np.random.uniform(3, 8)
        x.extend(np.random.normal(meanX, sigma, size))
        y.extend(np.random.normal(meanY, sigma, size))

    c = [None]*size*k
    df = pd.DataFrame({'c': c, 'x': x, 'y': y})
    #plt.scatter(x, y)
    #plt.show()
    #plt.clf()
    return df

def kMeans(df, k):

    m = pd.DataFrame({'x': [], 'y': [], 'c': []})
    x = df['x']
    y = df['y']
    newC = []

    centroidsX, centroidsY = getRandomCentroids(df, k)


    for i in range(len(x)):
        xval = df['x'][i]
        yval = df['y'][i]

        minDist = 100000
        selectedCluster = -1
        for j in range(k):
            cx = centroidsX[j]
            cy = centroidsY[j]

            dist = getDistance(cx, cy, xval, yval)
            if dist < minDist:
                selectedCluster = j
                minDist = dist

        cluster = "cluster" + str(selectedCluster)
        newC.append(selectedCluster)
        plt.plot(xval, yval, colours[selectedCluster]+".", markersize=1)

    df['c'] = newC

    for i in range(k):
        cluster = "cluster" + str(i)
        plt.plot(centroidsX[i], centroidsY[i], "k.", markersize=10)

   # plt.plot(x, y, '.', label=list(df['c']), alpha=0.5)
   # plt.legend()
    plt.show()

    Js.append(getJ(df, centroidsX, centroidsY))
    new_centroids = pd.DataFrame(df).groupby(by='c').mean().values
    #print(len(new_centroids))
    centroidsX = []
    centroidsY = []

    for i in range(k):
        centroidsX.append(new_centroids[i][0])
        centroidsY.append(new_centroids[i][1])


    for i in range(MAX_ITERATIONS):
        newC = []

        for i in range(len(x)):
            xval = df['x'][i]
            yval = df['y'][i]

            minDist = 100000
            selectedCluster = -1
            for j in range(k):
                cx = centroidsX[j]
                cy = centroidsY[j]

                dist = getDistance(cx, cy, xval, yval)
                if dist < minDist:
                    selectedCluster = j
                    minDist = dist

            cluster = "cluster" + str(selectedCluster)
            newC.append(selectedCluster)
            plt.plot(xval, yval, colours[selectedCluster] + ".", markersize=1)

        df['c'] = newC

        for i in range(k):
            cluster = "cluster" + str(i)
            plt.plot(centroidsX[i], centroidsY[i], "k.", markersize=10)

        # plt.plot(x, y, '.', label=list(df['c']), alpha=0.5)
        # plt.legend()
        plt.show()

        Js.append(getJ(df, centroidsX, centroidsY))
        new_centroids = pd.DataFrame(df).groupby(by='c').mean().values
        centroidsX = []
        centroidsY = []

        for i in range(k):
            centroidsX.append(new_centroids[i][0])
            centroidsY.append(new_centroids[i][1])


k = 4

df = generateData(k, 1000)
kMeans(df, k)

plt.clf()
plt.plot(range(MAX_ITERATIONS+1), Js)
plt.show()

#plt.scatter(x, y, alpha = 0.6, s=10)
#plt.show()




"""d_means = {'cluster 1': [0, 0],
           'cluster 2': [4, 5],
           'cluster 3': [5, 0]}
d_covs = {'cluster 1': [[1, 1],
                        [1, 4]],
          'cluster 2': [[1, 1],
                        [1, 3]],
          'cluster 3': [[4, 2],
                        [2, 2]]}





# Generate data based on the above parameters
n_data = 1000

# Generate data based on the above parameters
data = []
n_data = 1000
"""



"""
for cluster in d_means.keys():
    arr = np.random.multivariate_normal(d_means[cluster], d_covs[cluster], n_data)
    df_tmp = pd.DataFrame(arr)
    df_tmp['label'] = cluster
    l.append(df_tmp)
    plt.plot(df_tmp[0], df_tmp[1], '.', label=cluster, alpha=0.5)

plt.legend()
plt.axis('off')
plt.show()"""

"""features, clusters = make_blobs(n_samples=2000, n_features=2, centers=3, cluster_std=0.4, shuffle=True)

plt.scatter(features[:,0], features[:,1])

plt.show()"""