import pandas as pd
import os
import glob

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_GenerativeMTD.csv"))), ignore_index= True)
# df.to_csv('Results/class-GenerativeMTD-FinalResults.csv',index = False)


# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_TableGAN.csv"))), ignore_index= True)
# df.to_csv('Results/class-TableGAN-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_TVAE.csv"))), ignore_index= True)
# df.to_csv('Results/class-TVAE-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_VEEGAN.csv"))), ignore_index= True)
# df.to_csv('Results/class-VEEGAN-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_CTGAN.csv"))), ignore_index= True)
# df.to_csv('Results/class-CTGAN-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_class_*_CopulaGAN.csv"))), ignore_index= True)
# df.to_csv('Results/class-CopulaGAN-FinalResults.csv',index = False)

# df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "class-*-FinalResults.csv"))), ignore_index= True)
# df.to_csv('Results/class-GenerativeMTD-FinalResults.csv',index = False)






df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_GenerativeMTD.csv"))), ignore_index= True)
df.to_csv('Results/regress-GenerativeMTD-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_TableGAN.csv"))), ignore_index= True)
df.to_csv('Results/regress-TableGAN-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_TVAE.csv"))), ignore_index= True)
df.to_csv('Results/regress-TVAE-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_VEEGAN.csv"))), ignore_index= True)
df.to_csv('Results/regress-VEEGAN-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_CTGAN.csv"))), ignore_index= True)
df.to_csv('Results/regress-CTGAN-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "*_regress_*_CopulaGAN.csv"))), ignore_index= True)
df.to_csv('Results/regress-CopulaGAN-FinalResults.csv',index = False)

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join("Results", "regress-*-FinalResults.csv"))), ignore_index= True)
df.to_csv('Results/regress-GenerativeMTD-FinalResults.csv',index = False)

