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
def l2Norm(point,centroid):
    normVal = 0
    for i in range(len(point)):
        delta = point[i] - centroid[i]
        normVal += pow(delta,2)
    return normVal

def k_means_clustering(data_points, k, initial_centroids):
    for iteration in range(k):
        print("At iteration number : " + str(iteration))
        # number of iterations / training
        n = len(data_points)        #-of rows
        m = len(data_points[0])     #-of cols
        assignedCentroid = [-1 for idx in range(n)]
        zeroTuple = [0 for idx in range(m)]
        zeroCentroids = [zeroTuple for idx in range(k)]
        # [1] Assignment Step : cluster each observation to the nearest mean
        for point_idx, data_point in enumerate(data_points):
            # print("point idx = : " + str(point_idx))
            minimalDistance = float('inf')
            minimalIdx = -1
            for centroidIdx, centroid in enumerate(initial_centroids):
                curDistance = l2Norm(data_point,centroid)
                # print("l2 norm = : " + str(curDistance))
                if(curDistance <= minimalDistance):
                    minimalIdx = centroidIdx
                    minimalDistance = curDistance
            assignedCentroid[point_idx] = minimalIdx
        print(assignedCentroid)
        # [2] Update step : evolve our means based on the assignment of points to respective means
        nextCentroids = zeroCentroids
        for pointIdx, centroidIdx in enumerate(assignedCentroid):
            point = data_points[point_idx]
            curCentroidVal = nextCentroids[centroidIdx]
            nextVal = np.add(curCentroidVal, point)
            nextCentroids[centroidIdx] = nextVal
        initial_centroids = nextCentroids

    return assignedCentroid
