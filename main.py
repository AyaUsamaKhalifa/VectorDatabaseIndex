from vec_db import VecDB
import numpy as np

QUERY_SEED_NUMBER = 100
DB_SEED_NUMBER = 200

rng = np.random.default_rng(50)
vectors = rng.random((10**7*2, 70), dtype=np.float32)

rng = np.random.default_rng(QUERY_SEED_NUMBER)
query = rng.random((1, 70), dtype=np.float32)

actual_sorted_ids_20m = np.argsort(vectors.dot(query.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]