from ClusterMTD import *
from GenerativeMTD import *
import torch
from train_options import *
from gvae_data_transformer import *
import neptune
import neptune.new as neptune
from veegan import *
from veegan.veegan import VEEGAN
from tablegan import *
from tablegan.tablegan import TableGAN
import utils
from ast import literal_eval
from preprocess import find_cateorical_columns, match_dtypes
from ctgan import CTGANSynthesizer,TVAESynthesizer
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from pathlib import Path

def log_and_Save(real,fake,model_name,run,opt):
    result_df = pd.DataFrame()
    result_lst = []
    data_name = Path(opt.dataset).stem
    # data_name = str(opt.k) + "_" + Path(opt.dataset).stem
    # fake.to_csv('Data/Fake/fake_'+data_name+"_"+model_name+'.csv',index=False)

    pcd = utils.PCD(real,fake)
    run['output/Final PCD'] = pcd
    kstest, cstest = utils.stat_test(real,fake)
    run['output/KSTest'] = kstest
    run['output/CSTest'] = cstest

    dcr = utils.DCR(real,fake)
    nndr = utils.NNDR(real,fake)
    run['output/DCR'] = dcr
    run['output/NNDR'] = nndr

    if(opt.ml_utility == 'classification'):
        acc_tstr, f1_tstr = utils.predictive_model(real,fake,opt.class_col,'TSTR')
        run['output/Accuracy TSTR'] = acc_tstr
        run['output/F1 TSTR'] = f1_tstr

        acc_trts, f1_trts = utils.predictive_model(real,fake,opt.class_col,'TRTS')
        run['output/Accuracy TRTS'] = acc_trts
        run['output/F1 TRTS'] = f1_trts

        acc_trtr, f1_trtr = utils.predictive_model(real,real,opt.class_col,'TRTR')
        run['output/Accuracy TRTR'] = acc_trtr
        run['output/F1 TRTR'] = f1_trtr

        acc_tsts, f1_tsts = utils.predictive_model(match_dtypes(real,fake).copy(),match_dtypes(real,fake).copy(),opt.class_col,'TSTS')
        run['output/Accuracy TSTS'] = acc_tsts
        run['output/F1 TSTS'] = f1_tsts

        result_lst.append([data_name,model_name,opt.k,np.round(pcd,4),np.round(kstest,4),np.round(cstest,4),np.round(acc_tstr,4),np.round(f1_tstr,4),np.round(acc_trts,4),np.round(f1_trts,4),np.round(acc_trtr,4),np.round(f1_trtr,4),np.round(acc_tsts,4),np.round(f1_tsts,4),np.round(dcr,4),np.round(nndr,4)])
        result_df = pd.DataFrame(result_lst,columns=["DataName",'Method','k',"PCD","KSTest","CSTest", "Acc TSTR","F1 TSTR","Acc TRTS","F1 TRTS","Acc TRTR","F1 TRTR","Acc TSTS","F1 TSTS","DCR","NNDR"])
        result_df.to_csv('Results/'+ str(opt.k) + "_" + "class" + "_" +data_name + "_" + model_name + ".csv",index=False,float_format='%.4f')

    else:
        rmse_tstr, mape_tstr = utils.regression_model(real,fake,opt.class_col,'TSTR')
        run['output/RMSE TSTR'] = rmse_tstr
        run['output/MAPE TSTR'] = mape_tstr

        rmse_trts, mape_trts = utils.regression_model(real,fake,opt.class_col,'TRTS')
        run['output/RMSE TRTS'] = rmse_trts
        run['output/MAPE TRTS'] = mape_trts

        rmse_trtr, mape_trtr = utils.regression_model(real,real,opt.class_col,'TRTR')
        run['output/RMSE TRTR'] = rmse_trtr
        run['output/MAPE TRTR'] = mape_trtr

        rmse_tsts, mape_tsts = utils.regression_model(match_dtypes(real,fake).copy(),match_dtypes(real,fake).copy(),opt.class_col,'TSTS')
        run['output/RMSE TSTS'] = rmse_tsts
        run['output/MAPE TSTS'] = mape_tsts

        result_lst.append([data_name,model_name,opt.k,np.round(pcd,4),np.round(kstest,4),np.round(cstest,4),np.round(rmse_tstr,4),np.round(mape_tstr,4),np.round(rmse_trts,4),np.round(mape_trts,4),np.round(rmse_trtr,4),np.round(mape_trtr,4),np.round(rmse_tsts,4),np.round(mape_tsts,4),np.round(dcr,4),np.round(nndr,4)])
        result_df = pd.DataFrame(result_lst,columns=["DataName",'Method',"k","PCD","KSTest","CSTest", "RMSE TSTR","MAPE TSTR","RMSE TRTS","MAPE TRTS","RMSE TRTR","MAPE TRTR","RMSE TSTS","MAPE TSTS","DCR","NNDR"])
        result_df.to_csv('Results/' + str(opt.k) + "_" + "regress" + "_" +  data_name + "_" + model_name + ".csv",index=False,float_format='%.4f')
    
def flatten(lst, n):
    if n == 0:
        return lst
    return flatten([j for i in lst for j in i], n - 1)
    
def digitize_data(real,synthetic):
    temp_surr_data = pd.DataFrame(columns = list(real.columns))
    surrogate_data = pd.DataFrame(columns = list(real.columns))
    synth_data = pd.DataFrame(columns = list(real.columns))
    temp = pd.DataFrame(columns = list(real.columns))
    for col in real.columns:
        if(len(np.unique(real[[col]])) < 10): 
            bin_val =  np.unique(real[[col]])
            centers = (bin_val[1:]+bin_val[:-1])/2
            ind = np.digitize(synthetic[[col]].values, bins=centers , right=True)
            x = np.array([bin_val[i] for i in ind])
            x = np.array(flatten(x,1))
            print(x)
            temp[col] = pd.Series(x)  
        else:
            x = synthetic[[col]].values
            x = np.array(flatten(x,1))
            print(x)
            temp[col] = pd.Series(x)  
    temp_surr_data = pd.concat([temp_surr_data, temp])          
    surrogate_data = pd.concat([surrogate_data, temp_surr_data]) 
    return surrogate_data                         


def train_GenerativeMTD(opt):
    run = neptune.init(project="jaysivakumar/G-VAE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    # opt.dataname = Path(opt.dataset).stem
    model_name = "GenerativeMTD"
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = model_name
    run['config/criterion'] = "MMD + KLD + Cross entropy"
    run['config/optimizer'] = "SGD"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    opt.class_col = df.columns[opt.target_col_ix]
    run['config/class column'] = opt.class_col
    opt.cat_col = find_cateorical_columns(df)
    model = GenerativeMTD(opt, D_in,run)
    model.fit(df,discrete_columns = opt.cat_col)
    gvae_fake = model.sample(1000)
    gvae_fake = digitize_data(df,gvae_fake)
    run["digitized fake"].upload(File.as_html(gvae_fake))
    log_and_Save(df.copy(),gvae_fake.copy(),model_name,run,opt)
    run.stop()

def train_veegan(opt):
    run = neptune.init(project="jaysivakumar/VEEGAN", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    model_name = "VEEGAN"
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = model_name

    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    opt.class_col = df.columns[opt.target_col_ix]
    run['config/class column'] = opt.class_col
    opt.cat_col = find_cateorical_columns(df)
    model = VEEGAN(opt,run)
    model.fit(df,categorical_columns = opt.cat_col)
    veegan_fake = model.sample(1000)
    log_and_Save(df.copy(),veegan_fake.copy(),model_name,run,opt)
    run.stop()

def train_tablegan(opt):
    run = neptune.init(project="jaysivakumar/TableGAN", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    model_name = "TableGAN"
    run['config/model'] = model_name

    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    
    

    run["real data"].upload(File.as_html(df))

    opt.class_col = df.columns[opt.target_col_ix]
    run['config/class column'] = opt.class_col
    if(opt.dataname == 'breast'):
        df[[opt.class_col]] = df[[opt.class_col]].replace([2, 4], [0, 1])
    opt.cat_col = find_cateorical_columns(df)
    model = TableGAN(opt,run)
    model.fit(df,categorical_columns = opt.cat_col)
    tablegan_fake = model.sample(1000)
    if(opt.dataname == 'breast'):
        df[[opt.class_col]] = df[[opt.class_col]].replace([0, 1], [2, 4])
        tablegan_fake[[opt.class_col]] = tablegan_fake[[opt.class_col]].replace([0, 1], [2, 4])
    run["gen fake data"].upload(File.as_html(tablegan_fake))

    log_and_Save(df.copy(),tablegan_fake.copy(),model_name,run,opt)
    run.stop()


def train_ctgan(opt):
    run = neptune.init(project="jaysivakumar/CTGAN", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    model_name = "CTGAN"
    run['config/model'] = model_name

    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    opt.class_col = df.columns[opt.target_col_ix]
    run['config/class column'] = opt.class_col
    opt.cat_col = find_cateorical_columns(df)
    model = CTGANSynthesizer(epochs=opt.epochs)
    model.fit(df,discrete_columns = opt.cat_col)
    ctgan_fake = model.sample(1000)
    log_and_Save(df.copy(),ctgan_fake.copy(),model_name,run,opt)

    run.stop()


def train_copulagan(opt):
    run = neptune.init(project="jaysivakumar/CopulaGAN", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    model_name = "CopulaGAN"
    run['config/model'] = model_name

    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    opt.class_col = df.columns[opt.target_col_ix]
    opt.cat_col = find_cateorical_columns(df)
    run['config/class column'] = opt.class_col
    model = CopulaGAN(epochs=opt.epochs)
    model.fit(df)
    copulagan_fake = model.sample(num_rows=1000)
    print(copulagan_fake.isna().sum())
    log_and_Save(df.copy(),copulagan_fake.copy(),model_name,run,opt)
    run.stop()

def train_TVAE(opt):
    run = neptune.init(project="jaysivakumar/TVAE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset'] = Path(opt.dataset).stem
    opt.dataname = Path(opt.dataset).stem
    model_name = "TVAE"
    run['config/model'] = model_name

    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    opt.class_col = df.columns[opt.target_col_ix]
    run['config/class column'] = opt.class_col
    opt.cat_col = find_cateorical_columns(df)
    model = TVAESynthesizer(epochs=opt.epochs)
    model.fit(df,discrete_columns=opt.cat_col)
    tvae_fake = model.sample(1000)
    log_and_Save(df.copy(),tvae_fake.copy(),model_name,run,opt)
    run.stop()

if __name__ == "__main__":
    opt = TrainOptions().parse()
    print(opt)
    if(opt.model == 'veegan'):
        train_veegan(opt)
    if(opt.model == 'tablegan'):
        train_tablegan(opt)
    if(opt.model == 'ctgan'):
        train_ctgan(opt)
    if(opt.model == 'copulagan'):
        train_copulagan(opt)
    if(opt.model == 'TVAE'):
        train_TVAE(opt)
    if(opt.model == 'GenerativeMTD'):
        train_GenerativeMTD(opt)