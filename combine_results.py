import pandas as pd
import os
import glob

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*post_operative_GenerativeMTD.csv"))), ignore_index= True)
df.to_csv('Results/postop-GenerativeMTD-FinalResults.csv',index = False)


# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_TableGAN.csv"))), ignore_index= True)
# df.to_csv('Results/tablegan-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_TVAE.csv"))), ignore_index= True)
# df.to_csv('Results/TVAE-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_VEEGAN.csv"))), ignore_index= True)
# df.to_csv('Results/veegan-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_CTGAN.csv"))), ignore_index= True)
# df.to_csv('Results/ctgan-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_CopulaGAN.csv"))), ignore_index= True)
# df.to_csv('Results/copulagan-FinalResults.csv',index = False)

