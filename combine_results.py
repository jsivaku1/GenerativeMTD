import pandas as pd
import os
import glob

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_GenerativeMTD.csv"))), ignore_index= True)
df.to_csv('Results/FinalResults.csv',index = False)


