import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from torch import norm, topk
from torch.nn.modules.module import Module
from torch import tensor
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import torch


class kNNMTD(Module): 
    def __init__(self,opt):
        super().__init__()
        self.k = nn.Parameter(tensor(20.))
        self.opt = opt
        np.random.RandomState(self.opt.set_seed)
        self._gen_obs = self.opt.num_obs * 10
        
    def diffusion(self,sample):
        new_sample = []
        n = len(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2
        if(u_set == min_val or u_set == max_val):
            Nl = len([i for i in sample if i <= u_set])
            Nu = len([i for i in sample if i >= u_set])
        else:
            Nl = len([i for i in sample if i < u_set])
            Nu = len([i for i in sample if i > u_set])
        skew_l = Nl / (Nl + Nu)
        skew_u = Nu / (Nl + Nu)
        var = np.var(sample,ddof=1)
        if(var == 0):
            a = min_val/5
            b = max_val*5
            h=0
            new_sample = np.random.uniform(a, b, size=self._gen_obs) 
        else:
            h = var / n
            a = u_set - (skew_l * np.sqrt(-2 * (var/Nl) * np.log(10**(-20))))
            b = u_set + (skew_u * np.sqrt(-2 * (var/Nu) * np.log(10**(-20))))
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            while(len(new_sample) < self._gen_obs):
                    x = np.random.uniform(L,U)
                    if(x <= u_set):
                        MF = (x-L) / (u_set-L)
                    elif(x > u_set):
                        MF = (U-x)/(U-u_set)
                    elif(x < L or x > U) :
                        MF = 0
                    rs = np.random.uniform(0,1)
                    if(MF > rs):
                        new_sample.append(x)
                    else:
                        continue
        return a,b,np.array(new_sample)


    def findNeighbors(self, X, y):
        dist = norm(torch.from_numpy(X).view(1,-1) - torch.from_numpy(y).view(1,-1), dim=0,p=None)
        knn = dist.topk(self.opt.k+1, largest=False)
        return knn.indices.numpy()
    
    def flatten(self,lst, n):
        if n == 0:
            return lst
        return self.flatten([j for i in lst for j in i], n - 1)
    
    def generateData(self,train_array):
        synthSamples = []
        for ix, value in enumerate(train_array):
            temp_samples = []
            for col in range(train_array.shape[1]):
                X = np.array(train_array[:,col])
                y = np.array(value[col])
                indices = self.findNeighbors(X,y)
                array_to_diffuse = train_array[indices,col]
                L,U,new_sample = np.apply_along_axis(self.diffusion, 0, array_to_diffuse)
                temp_samples.append(new_sample.tolist())
            temp_samples = np.array(temp_samples).T.reshape(self._gen_obs,train_array.shape[1])
            synthSamples.append(temp_samples.tolist())
        synthSamples = np.array(self.flatten(synthSamples,1)).astype('float32')
        # df = pd.DataFrame(self._synthSamples) #convert to a dataframe
        
        return torch.from_numpy(synthSamples), synthSamples
