import pandas as pd
import os
import glob

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_GenerativeMTD.csv"))), ignore_index= True)
# df.to_csv('Results/GenerativeMTD-FinalResults.csv',index = False)


df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_tablegan.csv"))), ignore_index= True)
df.to_csv('Results/tablegan-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_TVAE.csv"))), ignore_index= True)
df.to_csv('Results/TVAE-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_veegan.csv"))), ignore_index= True)
df.to_csv('Results/veegan-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_ctgan.csv"))), ignore_index= True)
df.to_csv('Results/ctgan-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_copulagan.csv"))), ignore_index= True)
df.to_csv('Results/copulagan-FinalResults.csv',index = False)

