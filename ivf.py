##IVF Class

import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans,KMeans
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2
import os
import pickle
import gc
from sklearn import preprocessing
import math


class ivf:
    def load_next_batch(self, idx):
        filename = "batched_data/"+f'data_{idx}.pkl'
        if os.path.isfile(filename):
          file = open(filename, "rb")
          batch = pickle.load(file)
          return batch
        return False

    def append_to_file(self,file_name, id_, data_point):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id_] + data_point.tolist())

    def calc_cosine_similarity(self,vector1, vector2):
        dot_product = np.dot(vector1, vector2.T)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return 1-similarity

    def cleanup(self, path):
        # Get the list of all files and directories in the current working directory
        path = './' + path  # Current directory
        for file in os.listdir(path):
            if file.endswith('.pkl'):
                os.remove(os.path.join(path, file))
                print(f"Deleted file: {file}")

    def build_index(self,path,data = None):
     batch_size = 100000
     num_clusters = 4473
     index = [{} for _ in range(num_clusters)]
     centroids = None
     if not data:
      kmeans = MiniBatchKMeans(num_clusters,random_state=0,batch_size=batch_size,max_iter=10,n_init="auto")
      idx = 0
      while True:
        batch_dict = self.load_next_batch(idx)
        if (batch_dict == False):
          break
        batch_data = np.array(list(batch_dict.values()))
        for i in range(0, len(batch_data), batch_size):
          kmeans.partial_fit(preprocessing.normalize(batch_data[i:i+batch_size]))
          cluster_indices = kmeans.labels_
          for j,key in enumerate(cluster_indices):
           index[key][(idx*len(batch_data))+j+i] = batch_dict[(idx*len(batch_data))+j+i]
        del batch_dict
        del batch_data
        gc.collect()
        idx += 1
      centroids = kmeans.cluster_centers_

     else:
      data_without_ids = np.array(list(data.values()))
      num_clusters = math.ceil(math.sqrt(len(data_without_ids)))

      if(len(data_without_ids) > 1000000):
        kmeans = MiniBatchKMeans(num_clusters,random_state=0,batch_size=batch_size,max_iter=10,n_init="auto")
        for i in range(0, len(data_without_ids), batch_size):
          kmeans.partial_fit(preprocessing.normalize(data_without_ids[i:i+batch_size]))
          cluster_indices = kmeans.labels_
          for j,key in enumerate(cluster_indices):
           index[key][j+i] = data[j+i]
        centroids = kmeans.cluster_centers_

      else:
        centroids,cluster_indices = kmeans2(preprocessing.normalize(data_without_ids),num_clusters)
        for i,key in enumerate(cluster_indices):
           index[key][i] = data[i]

      del data
      del data_without_ids


     for i in range(num_clusters):
        # Define a filename for the cluster
        filename = path+f'cluster_{i}.pkl'
        file = open(filename,"wb")
        pickle.dump(index[i], file)
        file.close()
        print(f'Cluster {i} saved to {filename}')

     centroids_file_name = path+"centroids.pkl"
     file = open(centroids_file_name,"wb")
     pickle.dump(centroids, file)
     del index
     del centroids
     gc.collect()
     file.close()


    # Loading centroids from file
    def load_centroids(self,filename):
        file = open(filename, "rb")
        centroids = pickle.load(file)
        file.close()
        return centroids

    def load_points_from_file(self,file_name):
        file = open(file_name, "rb")
        points = pickle.load(file)
        file.close()
        return points

    def find_nearest_neighbors(self,path,query_point, centroids, k, n_centroids_to_consider=2):
        centroid_distances = self.calc_cosine_similarity([query_point], centroids)[0]
        nearest_centroids_indices = np.argsort(centroid_distances)[:n_centroids_to_consider]

        # Load points from nearest centroid files and calculate distances
        neighbor_candidates = []
        for idx in nearest_centroids_indices:
            points = self.load_points_from_file(path+f"cluster_{idx}.pkl")
            for point in points:
                point_vector = points[point]
                similarity = self.calc_cosine_similarity(point_vector, np.array(query_point))
                neighbor_candidates.append((point,points[point], similarity))

        # Sort by distance and select the nearest k
        neighbor_candidates.sort(key=lambda x: x[2])
        nearest_neighbors = [(x[0],x[1]) for x in neighbor_candidates[:k]]

        return nearest_neighbors