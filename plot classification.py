#%%
from numpy.lib.function_base import diff
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

diff_data = pd.concat([pd.read_csv('Results/class-GenerativeMTD-FinalResults.csv'),
                    pd.read_csv('Results/class-knnmtd-FinalResults.csv')])
# diff_data = diff_data.reset_index()
print(diff_data)

# plt.rcParams['axes.edgecolor'] = "0.01"
# plt.rcParams['axes.linewidth'] = 0.01
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 20, 12
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# sns.set_style("darkgrid", {"axes.facecolor": ".9"})

datasets = {
    'mammography': 'Mammographic Mass',
            'caesarian': 'Caesarean',
            'cryotherapy': 'Cryotherapy',
            'immunotherapy':'Immunotherapy',
            'post_operative': 'Post-operative',
            'breast': 'Wisconsin Breast Cancer',
            'cleveland_heart': 'Cleveland Heart Disease',
            'cervical': 'Cervical',
            'urban_land': 'Urban Land Cover',
            'imputed_SweatBinary': 'Sweat Study'}
# palette = sns.color_palette("Set1", 13)
diff_data = diff_data.replace({"DataName": datasets})

df = diff_data[diff_data.Method == 'GenerativeMTD']

fig, axes = plt.subplots(5,2, squeeze=True,figsize = (14, 16),frameon = True)

# plt.suptitle(r"PCD for benchmark datasets with different $\mathit{k}$ for $\mathit{k}$NNMTD")

for data,ax in zip(datasets.items(),axes.flat):
    label,name = data
    result = df[df.DataName==name]
    result['nndr_means'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[0]) for inner_list in result['NNDR']]
    result['nndr_std'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[1]) for inner_list in result['NNDR']]

    ax = sns.lineplot('k','PCD',color="#1f77b4",marker='o',data=result,ci=None,lw=1,ax=ax)
    
    # for k,pcd,nndr in result[['k','PCD','nndr_means']].values:
    #     label = "{:.2f}".format(nndr)
    #     ax.annotate(label, # this is the text
    #              (k,pcd), # these are the coordinates to position the label
    #              textcoords="offset points", # how to position the text
    #              xytext=(0,10), # distance from text to points (x,y)
    #              ha='center') # horizontal alignment can be left, right or center
        # ax.text(k,pcd,f'{nndr:.2f}',horizontalalignment='center',size='small')


    ax.set(xlabel=r'$\mathit{k}$', ylabel=r'PCD',title=r"{%s}" % name)
    ax.set(xlim=(2.9,10.1))
    ax.set_xticks([3,4,5,6,7,8,9,10])
    ax.autoscale()
    ax.legend([],[], frameon=False)
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    ax.get_yaxis().set_label_coords(-0.08,0.55)

    
# axes.flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig('/home/jay/Insync/jsivaku1@binghamton.edu/Google Drive/SS/Dissertation/Images/GenMTD-classification-sensitivity-diffdata.eps',dpi=500)
# fig.savefig('/Volumes/GoogleDrive/My Drive/SS/Dissertation/Images/GenMTD-classification-sensitivity-diffdata.eps',dpi=500)
plt.close('all')
plt.clf()


#%%


# %%

fig, axes = plt.subplots(5,2, squeeze=True,figsize = (19, 25),frameon = True)



for data,ax in zip(datasets.items(),axes.flat):
    label,name = data
    result = df[df.DataName==name].reset_index()
    result['nndr_means'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[0]) for inner_list in result['NNDR']]
    result['nndr_std'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[1]) for inner_list in result['NNDR']]
    ax = sns.lineplot('k','nndr_means',color="#1f77b4",style='Method',data=result,marker='o',ci='nndr_std', err_style='bars',lw=1,ax=ax)

    sns.despine()
    ax.set(xlabel=r'$\mathit{k}$', ylabel=r'NNDR',title=r"{%s}" % name)
    ax.set(xlim=(2.9,10.1),ylim=(0, 1))
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
# plt.savefig('/home/jay/Insync/jsivaku1@binghamton.edu/Google Drive/SS/Dissertation/Images/GenMTD-classification-priv.eps',dpi=500)
plt.show()
# plt.savefig('/Volumes/GoogleDrive/My Drive/SS/Dissertation/Images/GenMTD-classification-priv.eps',dpi=500)
plt.close('all')
plt.clf()
# %%
result
#%%

fig, axes = plt.subplots(5,2, squeeze=True,figsize = (19, 25),frameon = True)



for data,ax in zip(datasets.items(),axes.flat):
    label,name = data
    result = diff_data[(diff_data.DataName==name) & (diff_data.Method != 'kNNMTD')].reset_index()
    result['nndr_means'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[0]) for inner_list in result['NNDR']]
    result['nndr_std'] = [float(' '.join(inner_list.strip('][').split(' ')).split()[1]) for inner_list in result['NNDR']]
    ax = sns.boxplot('Method',r'nndr_means',data=result,ax=ax)

    # sns.despine()
    ax.set(xlabel=r'Method', ylabel=r'NNDR',title=r"{%s}" % name)
    # ax.set(xlim=(2.9,10.1),ylim=(0, 1))
    # ax.set_xticks([3,4,5,6,7,8,9,10])
    # ax.autoscale()
    # ax.legend([],[], frameon=False)
    # ax.spines['bottom'].set_color('0.5')
    # ax.spines['top'].set_color('0.5')
    # ax.spines['right'].set_color('0.5')
    # ax.spines['left'].set_color('0.5')
    # ax.get_yaxis().set_label_coords(-0.08,0.55)

    
axes.flat[-1].set_visible(False)
plt.tight_layout()
# plt.savefig('/home/jay/Insync/jsivaku1@binghamton.edu/Google Drive/SS/Dissertation/Images/GenMTD-classification-priv.eps',dpi=500)
plt.show()
# plt.savefig('/Volumes/GoogleDrive/My Drive/SS/Dissertation/Images/GenMTD-classification-priv.eps',dpi=500)
plt.close('all')
plt.clf()

# %%
