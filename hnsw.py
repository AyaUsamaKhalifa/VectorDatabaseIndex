from bisect import insort
import numpy as np
from heapq import heapify, heappop, heappush, nlargest
from operator import itemgetter
import copy

class hnsw:
    def __init__(self, m, mL, ef):
        self.L = 0
        self.mL = mL
        self.m = m
        self.m0 = 2 * m
        self.ef = ef
        self.entrypoint = None
        # list of L lists that represent the different levels
        # notice that if L = 3, we have levels 0,1&2
        # notice that L is the bottom level (the level that contains all entries)
        # each list will hold the node structure which consists of the vector, its neighbours, and the index of the current node in the next level (ana feen f level l+1?)
        self.data = dict()
        self.index = [] #List of dictionaries

    def __cosine_similarity(self,vec1, vec2):
      # the greater the number, the more similar the two vectors
      return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # insert the node into the index
    def insert(self,query):
      id, vec = query

      # nearest neighbours list
      W = []
      ep = [self.entrypoint]
      # else, get the level where we will insert the vector from the __getinsertlayer() function
      l = self.__getinsertlayer()
      #Add data to the dictionary
      self.data[id] = vec
      if self.entrypoint is not None:
        #for lc ← L … l+1
        for layer_num in reversed(range(l + 1, self.L)):
          W = self.__searchlayer(query, ep, 1, layer_num)
          ep = [t[1] for t in W]
        #for lc ← min(L, l) … 0
        for layer_num in reversed(range(0, min(l + 1, self.L))):
          m = self.m if layer_num != 0 else self.m0
          W = self.__searchlayer(query, ep, self.ef, layer_num)
          ep = [t[1] for t in W]
          # add bidirectionall connectionts from neighbors to q at layer lc
          self.index[layer_num][id] = dict()
          self.__select(id, W, layer_num, m)
          neigbours = copy.deepcopy(list(self.index[layer_num][id].items()))
          for key, dist in neigbours:
            self.__select(key, [(dist, id)], layer_num, m, update = True)

      for i in range(len(self.index), l+1):
        d = dict()
        d[id] = dict()
        self.index.append(d)
        self.L += 1
        self.entrypoint = id

    # search for the nearest K neighbours
    # K --> the number of nearest neighbours to be retrieved
    def search(self,query,ef,K):
      # loop over all levels
      # current nearest neighbours
      testquery = (0, query)
      W = []
      ep = [self.entrypoint]
      # from the top layer (0) to L-1 --> get the nearest neighbour
      for layer_num in reversed(range(1, self.L)):
        W = self.__searchlayer(testquery, ep, 1, layer_num)
        ep = [t[1] for t in W]
      # in the last level L --> get the nearest ef
      W = self.__searchlayer(testquery, ep, self.ef, 0)
      return W[-K:]


    # select the nearest neighbours in the level
    def __selectneighbours(self):
      print("select neighbours")

    # search for the nearest neighbours in a specific layer
    # query --> vector
    # entryPoints --> node
    def __searchlayer(self, query, entryPoints, ef, layer):
      similarity = [self.__cosine_similarity(self.data[entryPoints[i]], query[1]) for i in range(0, len(entryPoints))]
      # E7tmal akid 8alat
      # tuple consists of cosine similarity and the node
      C = [(similarity[i], entryPoints[i]) for i in range(0, len(entryPoints))]

      # set of visited elements V
      V = set(point for point in entryPoints)
      # set of candidates C
      heapify(C)
      # dynamic list of found nearest neighbors W
      W = list(C)
      # loop over C while it is not empty
      while C:
        # check if the most similar element in C is less similar than the least similar element in W --> break the loop (we won't find more similar vectors in C as it is sorted)
        # get the most similar element from C (highest value of cosine similarity)
        nearestC = C.pop()
        # get the least similar element from W (lowest value of cosine similarity)
        furthestW = W[0]
        # if cosinesimilarity(c, q) < cosinesimilarity(f, q) --> break
        if nearestC[0] < furthestW[0]:
          break
        # loop over all the neihbours of nearestC & update C & W
        # for each e ∈ neighbourhood(c) at layer lc // update C and W
        for e in self.index[layer][nearestC[1]].keys():
          # if e is not in Visited --> add
          if e not in V:
            V.add(e)
            # get the least similar element from W (lowest value of cosine similarity)
            furthestW = W[0]
            # if e is more similar than the least similar element in W or │W│ < ef
            e_similarity = self.__cosine_similarity(self.data[e], query[1]) #e[1]
            if (e_similarity > furthestW[0]) or (len(W) < ef):
              # add e to the candidates (C)
              heappush(C,(e_similarity,e))
              # W ← W ⋃ e
              insort(W, (e_similarity, e))
              # if │W│ > ef
              if len(W) > ef:
                # remove furthest element from W to q (remove the first element)
                W=W[1:]
      return W

    def __select(self, idx, candidates, layer, m, update = False):
      #Updating back links
      if update:
        if(len(self.index[layer][idx]) < m):
          self.index[layer][idx][candidates[0][1]] = candidates[0][0]
        else:
          min_idx = min(self.index[layer][idx].items(), key=itemgetter(1))
          if min_idx[1] < candidates[0][0]:
            del self.index[layer][idx][min_idx[0]]
            self.index[layer][idx][candidates[0][1]] = candidates[0][0]
          else:
            #To ensure bidirectional links
            del self.index[layer][candidates[0][1]][idx]
        return

      #m nearest elements to q
      to_insert = candidates[-m:]
      to_insert.reverse()
      #Insert new connections(neighbours)
      for inserted in to_insert:
        self.index[layer][idx][inserted[1]] = inserted[0]


    # get the index of the first layer where the node will be inserted
    def __getinsertlayer(self):
      # applying the equation l=int(-uniform(0,1)*mL)
      # an exponentially decaying function normalized by mL
      # it achieves inserting more elements in the lower levels than in higher ones
      return int(-1*np.log(np.random.random())*self.mL)