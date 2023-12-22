##IVF Class

import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans
import os
from scipy.cluster.vq import whiten, kmeans, vq, kmeans2


class ivf:
    def load_next_batch(self,file_handle, batch_size):
        batch_data = []
        batch_ids = []
        for _ in range(batch_size):
            line = file_handle.readline()
            if not line:
                break
            parts = line.strip().split(',')
            id_ = parts[0]
            vector = [float(x) for x in parts[1:]]
            batch_data.append(vector)
            batch_ids.append(id_)
        return batch_ids, np.array(batch_data)

    def append_to_file(self,file_name, id_, data_point):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id_] + data_point.tolist())

    def calc_cosine_similarity(self,vector1, vector2):
      dot_product = np.dot(vector1, vector2.T)
      norm_vector1 = np.linalg.norm(vector1)
      norm_vector2 = np.linalg.norm(vector2)
      similarity = dot_product / (norm_vector1 * norm_vector2)
      return similarity
      # dot_product = np.dot(vector1, vector2)
      # norm_vector1 = np.linalg.norm(vector1)
      # norm_vector2 = np.linalg.norm(vector2)
      # similarity = dot_product / (norm_vector1 * norm_vector2)
      # return similarity

    def cleanup(self):
        # Get the list of all files and directories in the current working directory
        path = '.'  # Current directory
        for file in os.listdir(path):
            if file.endswith('.csv'):
                os.remove(os.path.join(path, file))
                # print(f"Deleted file: {file}")

    def build_index(self,path,data):
        num_clusters = 4
        data_without_ids = np.array([d[1:] for d in data])
        # returns centroids and distortion
        centroids, _ = kmeans(data_without_ids, num_clusters)

        # Assign each sample to a cluster
        # returns cluster indices and distances
        cluster_indices, _ = vq(data_without_ids, centroids)
        print(cluster_indices == 1)

        for i in range(num_clusters):
            # Extract data points that belong to cluster i
            points_in_cluster = [d for j, d in enumerate(data) if cluster_indices[j] == i]

            # Define a filename for the cluster
            filename = f'cluster_{i}.csv'

            # Save the data points to the file
            np.savetxt(path+"/"+filename, points_in_cluster, delimiter=',', fmt='%f')

            print(f'Cluster {i} saved to {filename}')

        centroids_file_name = path+"/centroids.csv"
        np.savetxt(centroids_file_name, centroids, delimiter=',')


    # Loading centroids from file
    def load_centroids(self,filename):
        centroids = np.loadtxt(filename, delimiter=',')
        return centroids

    def load_points_from_file(seld,file_name):
        points = []
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                points.append([float(x) for x in row])
        return points

    def find_nearest_neighbors(self,path,query_point, centroids, k, n_centroids_to_consider=2):
        # Calculate distances to centroids and get indices of the nearest ones
        centroid_distances = self.calc_cosine_similarity([query_point], centroids)[0]
        # centroid_distances = distance.cdist([query_point], centroids, 'euclidean')[0]
        nearest_centroids_indices = np.argsort(centroid_distances)[:n_centroids_to_consider]

        # Load points from nearest centroid files and calculate distances
        neighbor_candidates = []
        for idx in nearest_centroids_indices:
            points = self.load_points_from_file(path+f"/centroid_{idx}.csv")
            for point in points:
                point_vector = np.array(point[1:])
                dist = np.linalg.norm(np.array(point_vector) - np.array(query_point))
                neighbor_candidates.append((point, dist))

        # Sort by distance and select the nearest k
        neighbor_candidates.sort(key=lambda x: x[1])
        nearest_neighbors = [x[0] for x in neighbor_candidates[:k]]

        return nearest_neighbors
