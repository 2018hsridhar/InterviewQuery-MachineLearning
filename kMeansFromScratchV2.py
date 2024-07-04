'''
Unsupervised Learning

URL := https://www.interviewquery.com/questions/k-means-from-scratch
k-Means From Scratch

Iterative processing algorithm
(A) It's an initial set of means -> we assign points and then update the means again
#-rows,#-cols is arbitrary ( take note ) for NumPy array
(B) We do not have a fixed number of iterations

For now, I will focus on just a single iteration
How many seperating lines in N-D space?

'''
import pandas as pd
import numpy as np

# assume that K := #-centroids
# Squared Euclidean distance desired ( guarantee convergence )

def l2Norm(point,centroid):
    normVal = 0
    for i in range(len(point)):
        delta = point[i] - centroid[i]
        normVal += pow(delta,2)
    return normVal

# The very number of centroids can also decrease too ( 0 points were assigned gaah ) 
def k_means_clustering(data_points, k, initial_centroids):
    iteration = 0
    while(True):
        # number of iterations / training
        n = len(data_points)        #-of rows
        m = len(data_points[0])     #-of cols
        assignedCentroid = [-1 for idx in range(n)]
        zeroTuple = [0 for idx in range(m)]
        zeroCentroids = [zeroTuple for idx in range(len(initial_centroids))]
        # [1] Assignment Step : cluster each observation to the nearest mean
        for point_idx, data_point in enumerate(data_points):
            # print("point idx = : " + str(point_idx))
            minimalDistance = float('inf')
            bestCentroidIdx = -1
            for centroidIdx, centroid in enumerate(initial_centroids):
                curDistance = l2Norm(data_point,centroid)
                # print("l2 norm = : " + str(curDistance))
                if(curDistance <= minimalDistance):
                    bestCentroidIdx = centroidIdx
                    minimalDistance = curDistance
            assignedCentroid[point_idx] = bestCentroidIdx
        # [2] Update step : evolve our means based on the assignment of points to respective means
        nextCentroids = zeroCentroids
        numPointsNextCentroids = [0 for i in range(len(initial_centroids))]
        for pointIdx, centroidIdx in enumerate(assignedCentroid):
            point = data_points[pointIdx]
            curCentroidVal = nextCentroids[centroidIdx]
            nextVal = np.add(curCentroidVal, point)
            nextCentroids[centroidIdx] = nextVal
            numPointsNextCentroids[centroidIdx] += 1
        # bug : if no points are assigned -> please chuck out that centroid ( clearly was an outlier )
        actualNextCentroids = []
        for idx in range(k):
            numPointsAssigned = numPointsNextCentroids[idx]
            if(numPointsAssigned >= 1):
                futureCentroid = nextCentroids[idx] / numPointsAssigned
                actualNextCentroids.append(futureCentroid)
        error = 0
        for i in range(len(initial_centroids)):
            numPointsAssigned = numPointsNextCentroids[idx]
            if(numPointsAssigned >= 1):
                delta = l2Norm(nextCentroids[i], initial_centroids[i])
                error += delta
        converged = (error == 0)
        if(converged):
            return assignedCentroid
        initial_centroids = actualNextCentroids
        iteration += 1
    return assignedCentroid
