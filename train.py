import ClusterMTD
from ClusterMTD import *
import VAE as VAE
import GVAE as GVAE
import AE as AE
import GAE as GAE

from VAE import *
from GVAE import *
from AE import *
from GAE import *
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import ClusterMTD
from ClusterMTD import *
from train_options import *
from data_transformer import *
import neptune
import neptune.new as neptune
import os
import sys
import argparse


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
    model = GVAE.GenerativeMTD(opt, D_in,run)
    model.fit_GVAE(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))

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
    model = GAE.GenerativeMTD(opt, D_in,run)
    model.fit_GAE(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))


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
    model = AE.GenerativeMTD(opt, D_in,run)
    model.fit_AE(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))


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
    model = VAE.GenerativeMTD(opt, D_in,run)
    model.fit_VAE(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))

if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 

    if(opt.model == 'GVAE'):
        train_GVAE(opt)
    if(opt.model == 'GAE'):
        train_GAE(opt)
    if(opt.model == 'VAE'):
        train_VAE(opt)
    if(opt.model == 'AE'):
        train_AE(opt)