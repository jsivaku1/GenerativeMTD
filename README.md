# GenerativeMTD: Synthetic Data Generation from Small Datasets Using Deep Generative Approach


## Usage 
```python3
import pandas as pd
import numpy as np
from GenerativeMTD import *
from utils import *
from gvae_data_transformer import *
from preprocess import find_cateorical_columns, match_dtypes

# Generate samples for unsupervised learning task
real = pd.read_csv('Data/wisconsin_breast.csv')
cat_col = find_cateorical_columns(real)
model = GenerativeMTD(real)
model.fit(df,discrete_columns = cat_col)
fake = model.sample(1000)
fake = digitize_data(df,fake)
```

