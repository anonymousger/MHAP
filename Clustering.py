# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:43:54 2020

@author: 
"""

from matplotlib import pyplot
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from kneed import KneeLocator
import numpy as np
from kneed import KneeLocator
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

class Clustering:
    def __init__(self,cluster_lists):
        self.cluster_lists = cluster_lists

    def scale_data(self,cluster_lists):
        print(len(cluster_lists[0]))
        for data in cluster_lists:
            for idx,seq in enumerate(data):
                if(seq != []):
                    max_seq = max(seq)
                    min_seq = min(seq)
                    i = 0
                    while (i < len(seq)):
                        seq[i] = (seq[i] - min_seq) / (max_seq - min_seq)
                        i += 1
                else:
                    data.pop(idx)
                    
        return cluster_lists
    
    def transform_to_same_length(self,data, max_length):
        n = len(data)   
        # the new set in ucr form np array
        trans_x = np.zeros((n, max_length), dtype=np.float64)
    
        # loop through each time series
        for i in range(n):
            mts = data[i]
            curr_length = len(mts)
            if(curr_length >= 4):
                idx = np.array(range(curr_length))
                idx_new = np.linspace(0, idx.max(), max_length)
                # linear interpolation
                f = interp1d(idx, mts, kind='cubic')
                new_ts = f(idx_new)
                trans_x[i] = new_ts
            
        return trans_x
    
    def k_mean_clustering(self,num_clusters,data):
        kmeans = KMeans(init="random",n_clusters=num_clusters,n_init=10,max_iter=300,random_state=42)
        kmeans.fit(data)
        kmeans_kwargs = {"init": "random","n_init": 12,"max_iter": 300,"random_state": 42,}
        # A list holds the SSE values for each k
        #print(kmeans.cluster_centers_)
        #for x in kmeans.cluster_centers_:
        #    pyplot.plot(x)
        #    pyplot.show()
        return kmeans.cluster_centers_

    def K_shape_clustering(self,num_clusters,data):
        #data = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(data)
        kshape = KShape(n_clusters=num_clusters, verbose=True, random_state=42)
        kshape.fit(data)
        return kshape.cluster_centers_
    
    def cluster_sequence_kmean(self,cluster_number):
        # scale the data between 0 and 1
        cluster_centers = []
        #cluster_lists = self.scale_data(self.cluster_lists)
        cluster_lists = self.cluster_lists
        #loop thriugh the periods for each CNN layer
        count = 0
        for layer in (cluster_lists):    
            #for idx,seq in enumerate(layer):
            #    if(len(seq) < 4):
            #        layer.pop(idx)
            #trans_x = self.transform_to_same_length(layer,max(map(len, layer)))
            #trans_x = np.nan_to_num(trans_x)
            #do clustering
            #layer = np.array(layer)
            cluster_centers.append(self.K_shape_clustering(cluster_number[count],layer))
            count +=1
        
        return  cluster_centers
