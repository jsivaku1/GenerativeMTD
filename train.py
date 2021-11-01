from ClusterMTD import *
from VAE import *
from GVAE import *
from AE import *
from GAE import *
from GenerativeMTD import *
import torch
from train_options import *
from data_transformer import *
import neptune
import neptune.new as neptune

def train_GMTD(opt):
    run = neptune.init(project="jaysivakumar/GMTD", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "G-MTD"
    run['config/criterion'] = "MMD"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    model = GMTD(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))



def train_GVAE(opt):
    run = neptune.init(project="jaysivakumar/G-VAE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "G-VAE"
    run['config/criterion'] = "MMD + CORAL + KL"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    model = GVAE(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))

def train_GAE(opt):
    run = neptune.init(project="jaysivakumar/G-AE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "G-AE"
    run['config/criterion'] = "MMD + CORAL + KL"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    model = GAE(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))


def train_AE(opt):
    run = neptune.init(project="jaysivakumar/AutoE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "AutoE"
    run['config/criterion'] = "MMD + CORAL + KL"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    model = AE(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))


def train_VAE(opt):
    run = neptune.init(project="jaysivakumar/VAE", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "VAE"
    run['config/criterion'] = "MMD + CORAL + KL"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object
    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    model = VAE(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))

if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 
    if(opt.model == 'GMTD'):
        train_GMTD(opt)
    if(opt.model == 'GVAE'):
        train_GVAE(opt)
    if(opt.model == 'GAE'):
        train_GAE(opt)
    if(opt.model == 'VAE'):
        train_VAE(opt)
    if(opt.model == 'AE'):
        train_AE(opt)