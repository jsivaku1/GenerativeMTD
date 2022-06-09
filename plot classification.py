
from numpy.lib.function_base import diff
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


diff_data = pd.read_csv('Results/GenerativeMTD classification FinalResults.csv')



plt.rcParams['axes.edgecolor'] = "0.01"
# plt.rcParams['axes.linewidth'] = 0.01
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 17, 8
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

datasets = {
    'mammo': 'Mammographic Mass',
            'caesarian': 'Caesarean',
            'cryo': 'Cryotherapy',
            'immuno':'Immunotherapy',
            'postop': 'Post-operative',
            'breast': 'Wisconsin Breast Cancer',
            'clev_heart': 'Cleveland Heart Disease',
            'cervical': 'Cervical',
            'urban_land': 'Urban Land Cover',
            'sweat': 'Sweat Study'}
# palette = sns.color_palette("Set1", 13)

df = diff_data.copy()

print(df.head())

df.reset_index(inplace=True,drop=True)
df = df.replace({"DataName": datasets})


fig, axes = plt.subplots(7,2, squeeze=True,figsize = (19, 25),frameon = True)

# plt.suptitle(r"PCD for benchmark datasets with different $\mathit{k}$ for $\mathit{k}$NNMTD")

for data,ax in zip(datasets.items(),axes.flat):
    label,name = data
    result = df[df.DataName==name]
    
    ax = sns.lineplot('k','PCD',color="black",marker='o',data=result,ci=None,lw=1,ax=ax)


    
    ax.set(xlabel=r'$\mathit{k}$', ylabel=r'PCD',title=r"{%s}" % name)
    ax.set(xlim=(2.9,10.1),ylim=(0, np.max(df['PCD'])))
    ax.set_xticks([3,4,5,6,7,8,9,10])
    ax.autoscale()
    ax.legend([],[], frameon=False)
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    ax.get_yaxis().set_label_coords(-0.08,0.55)

    
axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('../Figures/classification-sensitivity-diffdata.eps',dpi=500)
plt.close('all')
plt.clf()
