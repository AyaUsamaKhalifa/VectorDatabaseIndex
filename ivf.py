import numpy as np
import csv
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity


class ivf:
    def load_next_batch(self,file_handle, batch_size):
        batch_data = []
        batch_ids = []
        for _ in range(batch_size):
            line = file_handle.readline()
            if not line:
                break
            parts = line.strip().split(' ')
            id_ = parts[0]
            vector = [float(x) for x in parts[1:]]
            batch_data.append(vector)
            batch_ids.append(id_)
        return batch_ids, np.array(batch_data)

    def append_to_file(self,file_name, id_, data_point):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id_] + data_point.tolist())

    def build_index(self):
        # Initialize MiniBatchKMeans
        n_clusters = 5
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=5)

        # Open the file from which to load data
        with open("dataset.txt", 'r') as data_file:
            while True:
                batch_ids, batch_data = self.load_next_batch(data_file, 5)
                if batch_data.size == 0:
                    break  # No more data to read
                kmeans.partial_fit(batch_data)

                # Determine the nearest centroid for each point in the batch and save to the corresponding file
                for id_, point in zip(batch_ids, batch_data):
                    nearest_centroid_idx = np.argmin(distance.cdist([point], kmeans.cluster_centers_, 'euclidean'))
                    # nearest_centroid_idx = np.argmax(cosine_similarity([point], kmeans.cluster_centers_))
                    file_name = f"centroid_{nearest_centroid_idx}.csv"
                    self.append_to_file(file_name, id_, point)


        # Get the final centroids
        centroids = kmeans.cluster_centers_

        # Writing centroids to file
        centroids_file_name = "centroids.csv"
        np.savetxt(centroids_file_name, centroids, delimiter=',')

        # Output the centroids
        print("Final centroids:")
        print(centroids)

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

    def find_nearest_neighbors(self,query_point, centroids, k, n_centroids_to_consider=2):
        # Calculate distances to centroids and get indices of the nearest ones
        # centroid_distances = cosine_similarity([query_point], centroids)[0]
        centroid_distances = distance.cdist([query_point], centroids, 'euclidean')[0]
        nearest_centroids_indices = np.argsort(centroid_distances)[:n_centroids_to_consider]

        # Load points from nearest centroid files and calculate distances
        neighbor_candidates = []
        for idx in nearest_centroids_indices:
            points = self.load_points_from_file(f"centroid_{idx}.csv")
            for point in points:
                point_vector = np.array(point[1:])
                dist = np.linalg.norm(np.array(point_vector) - np.array(query_point))
                neighbor_candidates.append((point, dist))

        # Sort by distance and select the nearest k
        neighbor_candidates.sort(key=lambda x: x[1])
        nearest_neighbors = [x[0] for x in neighbor_candidates[:k]]

        return nearest_neighbors
