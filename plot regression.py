
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



cont_res = pd.read_csv('../Data/Final Results/regress-ContData-metrics.csv')
disc_res = pd.read_csv('../Data/Final Results/regress-DiscData-metrics.csv')
mix_res = pd.read_csv('../Data/Final Results/regress-Mixdata-metrics.csv')
diff_data = pd.read_csv('../Data/Final Results/regress-diffdata-metrics.csv')

        
#%%
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


cont_df = cont_res[cont_res["surr n_cols"] == 50][['DataName', 'Method', 'num_neighbors', 'surr n_cols', 'PCD', 'KSTest',
       'CSTest', 'RMSE TSTR', 'MAPE TSTR', 'RMSE TRTS', 'MAPE TRTS',
       'RMSE TRTR', 'MAPE TRTR', 'RMSE TSTS', 'MAPE TSTS', 'DCR', 'NNDR']]
disc_df = disc_res[disc_res["surr n_cols"] == 50][['DataName', 'Method', 'num_neighbors', 'surr n_cols', 'PCD', 'KSTest',
       'CSTest', 'RMSE TSTR', 'MAPE TSTR', 'RMSE TRTS', 'MAPE TRTS',
       'RMSE TRTR', 'MAPE TRTR', 'RMSE TSTS', 'MAPE TSTS', 'DCR', 'NNDR']]
mix_df = mix_res[mix_res["surr n_cols"] == 50][['DataName', 'Method', 'num_neighbors', 'surr n_cols', 'PCD', 'KSTest',
       'CSTest', 'RMSE TSTR', 'MAPE TSTR', 'RMSE TRTS', 'MAPE TRTS',
       'RMSE TRTR', 'MAPE TRTR', 'RMSE TSTS', 'MAPE TSTS', 'DCR', 'NNDR']]

diff_df = diff_data[['DataName', 'Method', 'num_neighbors', 'surr n_cols', 'PCD', 'KSTest',
       'CSTest', 'RMSE TSTR', 'MAPE TSTR', 'RMSE TRTS', 'MAPE TRTS',
       'RMSE TRTR', 'MAPE TRTR', 'RMSE TSTS', 'MAPE TSTS', 'DCR', 'NNDR']]

df = pd.concat([cont_df,disc_df,mix_df,diff_df])
df.reset_index(inplace=True,drop=True)
df = df.replace({"DataName": datasets})


fig, axes = plt.subplots(7,2, squeeze=True,figsize = (19, 25),frameon = True)

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
plt.savefig('../Figures/regression-sensitivity-diffdata.eps',dpi=500)
plt.close('all')
plt.clf()

#%%
# plt.rcParams['axes.edgecolor'] = "0.01"
plt.rcParams['axes.linewidth'] = 0.2
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = 10, 4
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
disc = disc_res[(disc_res.num_neighbors == 4)]
mix = mix_res[(mix_res.num_neighbors == 3)]
con = cont_res[(cont_res.num_neighbors == 3)]

palette = sns.color_palette("tab10", 4)

fig, ax = plt.subplots(1,3, sharey=True,frameon = True,figsize = (15, 6))
# plt.suptitle(r"Average PCD for regression task for Simulated Datasets")
ax[0] = sns.lineplot('surr n_cols','PCD',data=disc,hue='Method',ci=None,palette=palette,lw=1,ax=ax[0])
ax[0].set(xlabel=r'\# of Features', ylabel=r'Mean PCD',title=r'Discrete-only Data $(\mathit{k=4})$')
ax[0].legend([],[], frameon=False)
ax[0].spines['bottom'].set_color('0.5')
ax[0].spines['top'].set_color('0.5')
ax[0].spines['right'].set_color('0.5')
ax[0].spines['left'].set_color('0.5')

ax[1] = sns.lineplot('surr n_cols','PCD',data=con,hue='Method',ci=None,palette=palette,lw=1,ax=ax[1])
ax[1].set(xlabel=r'\# of Features', ylabel=r'Mean PCD',title=r'Continuous-only Data $(\mathit{k=3})$')
ax[1].legend([],[], frameon=False)
ax[1].spines['bottom'].set_color('0.5')
ax[1].spines['top'].set_color('0.5')
ax[1].spines['right'].set_color('0.5')
ax[1].spines['left'].set_color('0.5')

ax[2] = sns.lineplot('surr n_cols','PCD',data=mix, hue='Method',ci=None,palette=palette,lw=1,ax=ax[2])
ax[2].set(xlabel=r'\# of Features', ylabel=r'Mean PCD',title=r'Mixed Data $(\mathit{k=3})$')
ax[2].legend([],[], frameon=False)
ax[2].spines['bottom'].set_color('0.5')
ax[2].spines['top'].set_color('0.5')
ax[2].spines['right'].set_color('0.5')
ax[2].spines['left'].set_color('0.5')


handles, labels = ax[2].get_legend_handles_labels()
plt.tight_layout()
plt.figlegend(handles=handles[0:], labels=labels[0:],loc='center',borderaxespad=0., bbox_to_anchor=(0.5, -0.05), ncol=6)
# plt.subplots_adjust(top=0.85,right=0.9)
plt.savefig('../Figures/regression-simdata-numcols.eps',dpi=500)
plt.close('all')
plt.clf()


#%%
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# sweat = pd.read_csv("../Data/Classification Datasets/imputed_df.csv")
# sweat_ordinal = pd.read_csv("../Data/Regression Datasets/imputed_Sweat3.csv")
# pm_sweat_ordinal = pd.read_csv("../Data/Surrogate Data/sweat_ordinal_knnmtd.csv")
# smoteR_sweat = pd.read_csv("../Data/Surrogate Data/sweat_ordinal_smoteR.csv")
# ctgan_ordinal_sweat = pd.read_csv("../Data/Surrogate Data/sweat_ordinal_ctgan.csv")
# mtd_ordinal_sweat = pd.read_csv("../Data/Surrogate Data/sweat_ordinal_mtd.csv")

# plt.rcParams['figure.constrained_layout.use'] = False
# plt.rcParams['axes.edgecolor'] = "0.15"
# plt.rcParams['axes.linewidth'] = 1.00

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# sns.set_style("whitegrid", {'axes.grid' : False})
# sns.set_palette("Set1")

# def plot_func_reg(df1,df2,df3,label,col,ax):
#     sns.kdeplot(df1[col],label='Real',ax=ax)
#     sns.kdeplot(df2[col],label='Proposed',ax=ax)
#     sns.kdeplot(df3[col],label=label,ax=ax)
#     ax.set(xlabel=col, ylabel='')
#     ax.get_yaxis().set_label_coords(-0.4,0.5)
#     ax.legend([],[], frameon=False)
#     ax.spines['bottom'].set_color('0.5')
#     ax.spines['top'].set_color('0.5')
#     ax.spines['right'].set_color('0.5')
#     ax.spines['left'].set_color('0.5')

    
# fig, axs = plt.subplots(5, 4,figsize=(10,10),frameon=True,squeeze=True)
# # plt.suptitle("Density Plot")
# for ax,col in zip(axs.flat,sweat.columns):
#     plot_func_reg(sweat_ordinal,pm_sweat_ordinal,smoteR_sweat,"SmoteR",col,ax)
# handles, labels = ax.get_legend_handles_labels()
# fig.delaxes(axs[4,3]) 
# plt.subplots_adjust(right=0.99)
# plt.tight_layout()
# plt.legend(handles=handles, labels=labels,loc='lower center',borderaxespad=0., bbox_to_anchor=(0.5, -1.2), ncol=6)
# plt.savefig('../Figures/reg_smoter_density_plot.eps',bbox_inches='tight',dpi=500)





# fig, axs = plt.subplots(5, 4,figsize=(11,11),squeeze=True,frameon=True)
# # plt.suptitle("Density Plot")
# for ax,col in zip(axs.flat,sweat.columns):
#     plot_func_reg(sweat_ordinal,pm_sweat_ordinal,mtd_ordinal_sweat,"MTD",col,ax)
# handles, labels = ax.get_legend_handles_labels()
# fig.delaxes(axs[4,3]) 
# plt.subplots_adjust(right=0.99)

# plt.tight_layout()
# plt.legend(handles=handles, labels=labels,loc='lower center',borderaxespad=0., bbox_to_anchor=(0.5, -1.2), ncol=6)
# plt.savefig('../Figures/reg_mtd_density_plot.eps',bbox_inches='tight',dpi=500)




# fig2, axs2 = plt.subplots(5, 4,figsize=(11,11),squeeze=True,frameon=True)
# # plt.suptitle("Density Plot")
# for ax,col in zip(axs2.flat,sweat.columns):
#     plot_func_reg(sweat_ordinal,pm_sweat_ordinal,ctgan_ordinal_sweat,"CTGAN",col,ax)
# handles, labels = ax.get_legend_handles_labels()
# fig2.delaxes(axs2[4,3]) 
# plt.subplots_adjust(right=0.99)
# plt.tight_layout()
# plt.legend(handles=handles, labels=labels,loc='lower center',borderaxespad=0., bbox_to_anchor=(0.5, -1.2), ncol=6)
# plt.savefig('../Figures/reg_ctgan_density_plot.eps',bbox_inches='tight',dpi=500)

#%%
# k_lst = np.arange(3,11,1)
# fig, axes = plt.subplots(4,3, sharex=True,figsize = (15, 25))
# datasets = [{'label': 'thyroid','name': 'Thyroid',},
#             {'label': 'liver','name': 'Liver',},
#             {'label': 'pima','name': 'Pima',},
#             {'label': 'prostate','name': 'Prostate',},
#             {'label': 'fertility','name': 'Fertility',},
#             {'label': 'bio','name':'Bioconcentration',},
#             {'label': 'heartfail','name': 'Heart Failure',},
#             {'label': 'fat','name': 'Fat',},
#             {'label': 'parkinsons','name': "Parkinson's",},
#             {'label': 'community and crimes','name': 'Communities and Crimes',},
#             {'label': 'sweat-ordinal','name': "Sweat Study (Ordinal)",}]
            
            
# plt.suptitle(r"\textbf{PCD for benchmark datasets with different $\mathit{k}$")
# palette = sns.color_palette("Set1", 4)

# for data,ax in zip(datasets,axes.flat):
#     label,name = data['label'], data['name']
#     df = diff_data[(diff_data.DataName==label) & (diff_data.Method!='TVAE')]
#     ax = sns.lineplot('num_neighbors','Mean PCD',data=df,hue='Method',ci=None,palette=palette,ax=ax)
#     ax.set(xlabel=r'$\mathbf{\mathit{k}}$', ylabel=r'\textbf{Mean PCD}',title=r"\textbf{{%s}}" % name)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.set(ylim=(0, np.max(df['Mean PCD'])))
#     ax.set_xticks([3,4,5,6,7,8,9,10])
#     ax.autoscale()
#     ax.legend([],[], frameon=False)
# # ax.legend(handles=handles, labels=labels,loc="upper left")
# fig.tight_layout()
# axes.flat[-1].set_visible(False)
# fig.legend(handles=handles, labels=labels,loc='center', bbox_to_anchor=(0.5, -0.02), ncol=5)
# plt.subplots_adjust(top=0.92)
# plt.show()
# plt.savefig('../Figures/regression-sensitivity-diffdata.eps',bbox_inches='tight',dpi=500)


# # %%
# # %%

# %%