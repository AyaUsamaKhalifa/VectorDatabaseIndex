from typing import Dict, List, Annotated
import numpy as np
from ivf import ivf
import math

class VecDB:
    def __init__(self, file_path = "db_10k", new_db = True) -> None:
        self.file_path = file_path
        self.ivf_instance = ivf()
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path+"dataset.csv", "w") as fout:
                # if you need to add any head to the file
                pass

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]] = None):
        self.ivf_instance.cleanup(self.file_path)
        self.ivf_instance.build_index(self.file_path,rows)


    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        centroids = self.ivf_instance.load_centroids(self.file_path+"centroids.pkl")
        n_centroids_to_consider = 10
        app_data_size = math.ceil(math.pow(len(centroids),2))
        if app_data_size >= 1000000:
            n_centroids_to_consider = 30 + (app_data_size//1000000)
        nearest_neighbors = self.ivf_instance.find_nearest_neighbors(self.file_path,query[0],centroids,top_k,n_centroids_to_consider)
        return nearest_neighbors

