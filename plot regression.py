
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

diff_data = pd.read_csv('Results/GenerativeMTD regression FinalResults.csv')

        
plt.rcParams['axes.edgecolor'] = "0.01"
# plt.rcParams['axes.linewidth'] = 0.01
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 17, 8
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

datasets = {'thyroid':'Thyroid',
            'liver': 'Liver',
            'pima': 'Pima',
            'prostate': 'Prostate',
            'fertility': 'Fertility',
            'bioconcentration':'Bioconcentration',
            'heartfail': 'Heart Failure',
            'fat': 'Fat',
            'parkinsons': "Parkinson's",
            'community_crime': 'Communities and Crimes',
            'sweat_ordinal': "Sweat Study (Ordinal)"}


df = diff_data.copy()
df.reset_index(inplace=True,drop=True)
df = df.replace({"DataName": datasets})


fig, axes = plt.subplots(5,2, squeeze=True,figsize = (19, 25),frameon = True)

# plt.suptitle(r"PCD for benchmark datasets with different $\mathit{k}$ for $\mathit{k}$NNMTD")

for data,ax in zip(datasets.items(),axes.flat):
    label,name = data
    result = df[(df.DataName==name) & (df.Method=='kNNMTD')]
    ax = sns.lineplot('num_neighbors','PCD',color="black",marker="o",data=result,ci=None,lw=1,ax=ax)
    ax.set(xlabel=r'$\mathit{k}$', ylabel=r'Mean PCD',title=r"{%s}" % name)
    ax.set(xlim=(2.9,10.1),ylim=(0, np.max(df['PCD'])))
    ax.set_xticks([3,4,5,6,7,8,9,10])
    ax.autoscale()
    ax.legend([],[], frameon=False)
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    ax.get_yaxis().set_label_coords(-0.08,0.55)

plt.tight_layout()
plt.savefig('Figures/GenMTD-regression-sensitivity-diffdata.eps',dpi=500)
