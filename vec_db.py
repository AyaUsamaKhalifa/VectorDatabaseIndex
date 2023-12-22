from typing import Dict, List, Annotated
import numpy as np
from ivf import ivf

class VecDB:
    def __init__(self, file_path = "db_10k", new_db = True) -> None:
        self.file_path = file_path
        self.ivf_instance = ivf()
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path+"dataset.csv", "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # with open(self.file_path+"/dataset.csv", "a+") as fout:
        #     for row in rows:
        #         id, embed = row["id"], row["embed"]
        #         row_str = f"{id}," + ",".join([str(e) for e in embed])
        #         fout.write(f"{row_str}\n")
        # self.ivf_instance.cleanup()
        modified_vectors = []

        # Iterate over each row and modify the vector
        for row in rows:
            id, embed = row["id"], row["embed"]
            # Prepend the id to the vector
            modified_vector = [id] + embed
            modified_vectors.append(modified_vector)

        self.ivf_instance.build_index(self.file_path,modified_vectors)

    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        centroids = self.ivf_instance.load_centroids(self.file_path+"centroids.csv")
        # query_point = np.array([0.35270562538935846, 0.5157003735414303, 0.46564154423765514, 0.2763390901691064, 0.5190477443622806, 0.9790123488797479, 0.18448167694248352, 0.5279077264012795, 0.4382358716000273, 0.06426522061686546, 0.32233119774613006, 0.7222802877207771, 0.7257624862300537, 0.933998471115222, 0.926839291511224, 0.4106540029875969, 0.5244840890821805, 0.4923512075281917, 0.6060284908221644, 0.49145078571165557, 0.7075774690092598, 0.6208347911529841, 0.38440258339576583, 0.057846137076901005, 0.27633647341742984, 0.2558303777580996, 0.5532732552871744, 0.06105580227635998, 0.8360758210324999, 0.25598750149330574, 0.6527336302555207, 0.9176501457788113, 0.9093738708792614, 0.46287135335635676, 0.5931334094919858, 0.49064527388638657, 0.5062097802510552, 0.907293472610603, 0.4650134970873995, 0.43755768237559656, 0.3563017316288559, 0.5842891088319624, 0.7824370394419798, 0.3017383193657802, 0.4538408084198833, 0.6689570696642663, 0.6490023852613068, 0.6391624289603409, 0.4989025325354879, 0.2479956193667734, 0.3455148817118454, 0.29728789797594246, 0.6554543874897971, 0.10345848925869872, 0.08518489652408323, 0.17893259877563417, 0.1696489039303003, 0.7235076610675364, 0.557379890276246, 0.7682546746759075, 0.001294493080513437, 0.4118253273681436, 0.9073449495488528, 0.8333947341500777, 0.6230473258972887, 0.0750128871504051, 0.1588579443339604, 0.6228405411536673, 0.21387947384297024, 0.3717030660431313])
        nearest_neighbors = self.ivf_instance.find_nearest_neighbors(self.file_path,query[0],centroids,top_k,3)
        return [int(s[0]) for s in nearest_neighbors]

