import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
import math

warnings.filterwarnings("ignore")

colours = ["b", "g", "r", "c", "m", "y", "k", "w"]
colors = ['blue', 'red', 'green', 'yellow', 'purple', 'darkblue', 'pink', 'orange']

MAX_ITERATIONS = 15
Js = []


def calculateWCSS(df, k):
    centroidsX, centroidsY = kMeans(df, k)
    return getJ(df, centroidsX, centroidsY)


def elbowMethod(df):
    w = []
    for k in range(1, 9):
        w.append(calculateWCSS(df, k))

    plt.clf()
    plt.plot(range(1, 9), w)
    plt.show()
    plt.clf()


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
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def getRandomCentroids(df, k):
    x = df['x']
    y = df['y']
    selectedX = random.sample(list(x), k)
    selectedY = random.sample(list(y), k)

    return selectedX, selectedY


def linearFunc(x, m, b):
     return m*x + b

def generateDifficultData(size):

    X = np.linspace(0, np.random.uniform(5, 10), size)

    above = [linearFunc(x, 10, 5) + abs(np.random.normal(20, 5)) for x in X]
    below = [linearFunc(x, 10, 5) - abs(np.random.normal(20, 5)) for x in X]

    above.extend(below)

    Y = above
    X = list(X)
    X.extend(X)
    c = ["cluster 0"] * len(Y)
    df = pd.DataFrame({'c': c, 'x': X, 'y': Y})
    return df


def generateData(k, size):
    x = []
    y = []

    for cluster in range(k):
        meanX = np.random.uniform(0, k * 15)
        meanY = np.random.uniform(0, k * 15)
        sigma = np.random.uniform(3, 8)
        x.extend(np.random.normal(meanX, sigma, size))
        y.extend(np.random.normal(meanY, sigma, size))

    c = [None] * size * k
    df = pd.DataFrame({'c': c, 'x': x, 'y': y})
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
        plt.plot(xval, yval, colours[selectedCluster] + ".", markersize=1)

    df['c'] = newC

    for i in range(k):
        cluster = "cluster" + str(i)
        plt.plot(centroidsX[i], centroidsY[i], "k.", markersize=10)

    plt.title("Iteration 1")
    plt.show()

    Js.append(getJ(df, centroidsX, centroidsY))
    new_centroids = pd.DataFrame(df).groupby(by='c').mean().values
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

        plt.show()

        Js.append(getJ(df, centroidsX, centroidsY))
        new_centroids = pd.DataFrame(df).groupby(by='c').mean().values
        oldCentroidsX = centroidsX
        oldCentroidsY = centroidsY
        centroidsX = []
        centroidsY = []

        for i in range(k):
            centroidsX.append(new_centroids[i][0])
            centroidsY.append(new_centroids[i][1])

        distances = []

        for i in range(k):
            distances.append(getDistance(centroidsX[i], centroidsY[i], oldCentroidsX[i], oldCentroidsY[i]))

        meanDist = sum(distances)/k
        #print(meanDist)
        if meanDist <= 0.01:
            return centroidsX, centroidsY

    return centroidsX, centroidsY


k = 3

df = generateData(k, 1000)
#df = generateDifficultData(500)

plt.scatter(df['x'], df['y'], s=0.5)
plt.title("Dataset")
plt.show()
plt.clf()

centroidsX, centroidsY = kMeans(df, k)
centers = []

for i in range(k):
    centers.append([centroidsX[i], centroidsY[i]])

plt.clf()
plt.plot(range(len(Js)), Js)
plt.show()

kmeans = KMeans(n_clusters=k, init="k-means++")
kmeans = kmeans.fit(df[['x', 'y']])

print("Found centroids using my method:")
print(centers)
print("Found centroids using sklearn library:")
print(kmeans.cluster_centers_)

#elbowMethod(df)
