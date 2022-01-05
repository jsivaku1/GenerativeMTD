import pandas as pd
import numpy as np
import faraway.datasets.hips
import faraway.datasets.prostate
import faraway.datasets.pima
import faraway.datasets.fat
from sklearn.preprocessing import LabelEncoder
import os


mydir = 'Data/Classification Datasets'
sweat = pd.read_csv(os.path.join(mydir,"imputed_df.csv"))
immuno = pd.read_csv(os.path.join(mydir,'Immunotherapy.csv'))
cryo = pd.read_csv(os.path.join(mydir,'Cryotherapy.csv'))
caesarian = pd.read_csv(os.path.join(mydir,'caesarian.csv'))
cervical = pd.read_csv(os.path.join(mydir,'cervical.csv'))



breast = pd.read_csv(os.path.join(mydir,'breast-cancer-wisconsin-diag.data'),header=None)
breast.drop(0,axis=1,inplace=True)
breast.replace({'?':np.nan},inplace=True)
breast.dropna(axis=0,inplace=True)
breast.reset_index(inplace=True, drop=True)
breast = breast.astype(int)

postop = pd.read_csv(os.path.join(mydir,'post-operative.csv'),header=None)
postop.replace({'?':np.nan},inplace=True)
postop.dropna(axis=0,inplace=True)
postop.reset_index(inplace=True, drop=True)
le = LabelEncoder()
postop[7] = postop[7].astype(int)
postop[[0,1,2,3,4,5,6,8]] = postop[[0,1,2,3,4,5,6,8]].apply(LabelEncoder().fit_transform)

mammo = pd.read_csv(os.path.join(mydir,'mammographic_masses.data'),header=None)
mammo.replace({'?':np.nan},inplace=True)
mammo.dropna(axis=0,inplace=True)
mammo.reset_index(inplace=True, drop=True)
mammo = mammo.astype(int)


clev_heart = pd.read_csv(os.path.join(mydir,'processed.cleveland.data'),header=None)
clev_heart.replace({'?':np.nan},inplace=True)
clev_heart.dropna(axis=0,inplace=True)
clev_heart.reset_index(inplace=True, drop=True)
clev_heart = clev_heart.astype(float)
for col in [1,2,5]:
    clev_heart[col] = clev_heart.astype(int)
    
urban_land = pd.read_csv(os.path.join(mydir,'urban-land-cover.csv'))
urban_land.replace({'?':np.nan},inplace=True)
urban_land.dropna(axis=0,inplace=True)
urban_land.reset_index(inplace=True, drop=True)
urban_land[["class"]] = urban_land[["class"]].apply(LabelEncoder().fit_transform)

datasets = [{'label': 'clev_heart','dataset': clev_heart,'class':clev_heart.columns[-1],},
            {'label': 'urban_land','dataset': urban_land,'class':'class',},
            {'label': 'mammo','dataset': mammo,'class':mammo.columns[-1],},
            {'label': 'sweat','dataset': sweat,'class':'PsychDistress',},
            {'label': 'immuno','dataset': immuno,'class':'Result_of_Treatment',},
            {'label': 'cryo','dataset': cryo,'class':'Result_of_Treatment',},
            {'label': 'caesarian','dataset': caesarian,'class':'Caesarian',},
            {'label': 'cervical','dataset': cervical,'class':'ca_cervix',},
            {'label': 'breast','dataset': breast,'class':breast.columns[-1],},
            {'label': 'postop','dataset': postop,'class':postop.columns[-1],}]

clev_heart.to_csv('Data/cleveland_heart.csv',index=False)
urban_land.to_csv('Data/urban_land.csv',index=False)
mammo.to_csv('Data/mammography.csv',index=False)
sweat.to_csv('Data/sweat_binary.csv',index=False)
immuno.to_csv('Data/immunotherapy.csv',index=False)
cryo.to_csv('Data/cryotherapy.csv',index=False)
caesarian.to_csv('Data/caesarian.csv',index=False)
cervical.to_csv('Data/cervical.csv',index=False)
breast.to_csv('Data/breast.csv',index=False)
postop.to_csv('Data/post_operative.csv',index=False)


sweat_ordinal = pd.read_csv("Data/Regression Datasets/imputed_Sweat3.csv")

parkinsons = pd.read_csv("Data/Regression Datasets/parkinsons.csv")
parkinsons.drop(["name"],axis=1,inplace=True)
#parkinsons.replace({'?':np.nan},inplace=True)
parkinsons.dropna(axis=0,inplace=True)
parkinsons.reset_index(inplace=True, drop=True)

thyroid = pd.read_csv("Data/Regression Datasets/thyroid.csv",header=None)
#thyroid.replace({'?':np.nan},inplace=True)
thyroid.dropna(axis=0,inplace=True)
thyroid.reset_index(inplace=True, drop=True)

liver = pd.read_csv("Data/Regression Datasets/bupa.csv", header = None)
liver.drop([6],axis=1,inplace=True)
#bupa.replace({'?':np.nan},inplace=True)
liver.dropna(axis=0,inplace=True)
liver.reset_index(inplace=True, drop=True)

bio = pd.read_csv("Data/Regression Datasets/bioconcentration.csv")
bio.drop(["CAS","SMILES","Set"],axis=1,inplace=True)
#bio.replace({'?':np.nan},inplace=True)
bio.dropna(axis=0,inplace=True)
bio.reset_index(inplace=True, drop=True)

fertility = pd.read_csv('Data/Regression Datasets/fertility_Diagnosis.csv',header=None)
fertility.replace({'?':np.nan},inplace=True)
fertility.dropna(axis=0,inplace=True)
fertility.reset_index(inplace=True, drop=True)
le = LabelEncoder()
fertility[[9]] = fertility[[9]].apply(LabelEncoder().fit_transform)



heartfail = pd.read_csv('Data/Regression Datasets/S1Data.csv')
#heartfail.replace({'?':np.nan},inplace=True)
heartfail.dropna(axis=0,inplace=True)
heartfail.reset_index(inplace=True, drop=True)

ccn = pd.read_csv('Data/Regression Datasets/community-crimes-normalized.csv')
ccn.drop(['state','county','community','communityname','fold'],axis=1,inplace=True)
ccn.replace({'?':np.nan},inplace=True)
ccn.dropna(axis=0,inplace=True)
ccn.reset_index(inplace=True, drop=True)
ccn = ccn.apply(pd.to_numeric) # convert all columns of DataFrame

fat = faraway.datasets.fat.load()
fat.drop(["siri"],axis=1,inplace=True)

prostate = faraway.datasets.prostate.load()

hips = faraway.datasets.hips.load()
hips.drop(["person"],axis=1,inplace=True)
hips[["grp","side"]] = hips[["grp","side"]].apply(LabelEncoder().fit_transform)

pima = faraway.datasets.pima.load()

datasets = [
            {'label': 'community and crimes','dataset': ccn,'class':ccn.columns[-1],},
            {'label': 'sweat-ordinal','dataset': sweat_ordinal,'class':"PsychDistress",},
            {'label': 'fertility','dataset': fertility,'class':fertility.columns[-2],},
            {'label': 'parkinsons','dataset': parkinsons,'class':"PPE",},
            {'label': 'thyroid','dataset': thyroid,'class':thyroid.columns[-1],},
            {'label': 'liver','dataset': liver,'class':liver.columns[1],},
            {'label': 'fat','dataset': fat,'class':'brozek',},
            {'label': 'pima','dataset': pima,'class':'diabetes',},
            {'label': 'prostate','dataset': prostate,'class':'lpsa',},
            {'label': 'bio','dataset': bio,'class':'logBCF',},
            {'label': 'heartfail','dataset': heartfail,'class':'CPK',}]

ccn.to_csv('Data/community_crime.csv',index=False)
sweat_ordinal.to_csv('Data/sweat_ordinal.csv',index=False)
fertility.to_csv('Data/fertility.csv',index=False)
parkinsons.to_csv('Data/parkinsons.csv',index=False)
thyroid.to_csv('Data/thyroid.csv',index=False)
liver.to_csv('Data/liver.csv',index=False)
fat.to_csv('Data/fat.csv',index=False)
pima.to_csv('Data/pima.csv',index=False)
prostate.to_csv('Data/prostate.csv',index=False)
bio.to_csv('Data/bioconcentration.csv',index=False)
heartfail.to_csv('Data/heartfail.csv',index=False)