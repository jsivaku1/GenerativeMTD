#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#%%
m = nn.LeakyReLU(0.1)
input = torch.randn(2)
output = m(input)
print(input)
output
#%%
df = pd.read_csv('testfile')
df.describe()

df.tail()
# %%
# Creating the data
df = pd.read_csv(os.path.join('../Data', "imputed_SweatBinary.csv"))

X = df.to_numpy()
# df.head()
# # Visualizing the data
# plt.plot()
# # plt.xlim([0, 10])
# # plt.ylim([0, 10])
# plt.title('Dataset')
# plt.scatter(X[:,0],X[:,1])
# plt.show()


#%%
data = torch.randn(1, 5)
test = data[ :, 0]

dist = torch.norm(data - test, dim=0, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices.numpy()))


# %%
data

#%%
test


# %%

def __init__(self,train,opt):
    super().__init__()
    self.k = nn.Parameter(tensor(20.))
    self.train = train
    self.column_names = train.columns
    self.train_array = train.to_numpy()
    self.opt = opt
    np.random.RandomState(self.opt.set_seed)
    self._gen_obs = self.opt.num_obs * 10
    self._synthSamples = np.zeros((0,5))     
    self.surrogate_data = pd.DataFrame()

def diffusion(sample):
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
        new_sample = np.random.uniform(a, b, size=2) 
    else:
        h = var / n
        a = u_set - (skew_l * np.sqrt(-2 * (var/Nl) * np.log(10**(-20))))
        b = u_set + (skew_u * np.sqrt(-2 * (var/Nu) * np.log(10**(-20))))
        L = a if a <= min_val else min_val
        U = b if b >= max_val else max_val
        while(len(new_sample) < 2):
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

def findNeighbors(X, y):


    dist = torch.norm(torch.from_numpy(X).view(1,-1) - torch.from_numpy(y).view(1,-1), dim=0,p=None)
    knn = dist.topk(5, largest=False)
    return knn.indices.numpy()


#%%



def generateData(train_array):
    _synthSamples = []
    temp_samples = []
    for ix, value in enumerate(train_array):
        print(type(value))
        if(ix==3):
            break
        temp_samples = []
        for col in range(train_array.shape[1]):

            X = np.array(train_array[:,col])
            y = np.array(value[col])
            indices = findNeighbors(X,y)
            array_to_diffuse = train_array[indices,col]
            print(temp_samples)
            L,U,new_sample = np.apply_along_axis(diffusion, 0, array_to_diffuse)
            temp_samples.append(new_sample.tolist())
        print(np.array(temp_samples).T.reshape(2,train_array.shape[1]))
        _synthSamples.append(np.array(temp_samples).T.reshape(2,train_array.shape[1]))
    _synthSamples = np.array(flatten(_synthSamples,1))
    print(_synthSamples)
    print(_synthSamples.shape)
    return torch.from_numpy(_synthSamples)

generateData(df.to_numpy())
# %%
a = [1,2,3]
# %%
np.array(a)
# %%
X
# %%
