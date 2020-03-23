import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import encoding_loader

class KClustering:
    
    
    def __init__(self, enc_dir='./encodings121', n_clusters=100):
        self.enc_dir = enc_dir
        self.n_clusters = n_clusters
        self.batch_to_index_to_filename = {}
        self.enc_matrix = self.combineEncodings()
        self.centers = []
        
    #Combines encoding files into one giant numpy array. Each row = an encoding
    def combineEncodings(self):
        enc_loader = encoding_loader.EncodingLoader(self.enc_dir)
        enc_matrix, index_to_filename = enc_loader.load_encodings_from_file('enc_0.npy', as_dict=False)
        batch_to_index_to_filename = {}
        batch_to_index_to_filename[0] = index_to_filename
        for i in range(1,97): #1,97
            curr_matrix, curr_filenames = enc_loader.load_encodings_from_file('enc_' + str(i) + '.npy', as_dict=False)
            enc_matrix = np.vstack((enc_matrix , curr_matrix))
            batch_to_index_to_filename[i] = curr_filenames
        self.batch_to_index_to_filename = batch_to_index_to_filename
        return enc_matrix
    
    def encodingMatrix(self):
        return self.enc_matrix
    
    def createClusters(self):
        km = KMeans(
            n_clusters=self.n_clusters, init='random',
            n_init=10, max_iter=100,
            tol=1e-04, random_state=0
        )
        
        y_km = km.fit_predict(self.enc_matrix)
        self.centers = km.cluster_centers_
        np.save('./ClusterCenters/centers', km.cluster_centers_)
       
        return y_km
        
    # Returns dictionary of clusters to the encodings that comprise them
    def clusters_to_encodings(self):
        y_km = self.createClusters()
        group_to_enc = {}
        for i in range(100):
            bool_arr = (y_km == i)
            result = np.where(bool_arr)
            group_to_enc[i] = result
        np.save('./ClusterCenters/cluster_to_encodings_dict', group_to_enc)
        return group_to_enc
        

        
