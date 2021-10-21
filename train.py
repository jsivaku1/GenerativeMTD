import ClusterMTD
from ClusterMTD import *
import models
from models import *
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
import models
from ClusterMTD import *
from models import *
from train_options import *
from data_transformer import *
import neptune
import neptune.new as neptune
import os
import sys
import argparse

if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 

    run = neptune.init(project="jaysivakumar/VAE-MTD", api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZTE3OWZiNS0xNzkyLTQ0ZjYtYmVjMC1hOWE1NjE4MGQ3MzcifQ==')  # your credentials
    run['config/dataset/path'] = opt.file
    # run['config/dataset/transforms'] = data_tfms # dict() object
    # run['config/dataset/size'] = dataset_size # dict() object
    run['config/model'] = "VAE-MTD"
    run['config/criterion'] = "MMD + CORAL + KL"
    run['config/optimizer'] = "Adam"
    # run['config/params'] = hparams # dict() object


    # By default, read csv file in the same directory as this script
        


    data = LoadFile(opt,run)
    D_in = data.__dim__()
    df,opt = data.load_data()
    


    # synth_df = kNNMTD(opt,df,run)
    # synthData_MTD = synth_df.generateData()
    # Call the main function of the script



    model = GenerativeMTD(opt, D_in,run)
    model.fit(df,discrete_columns = ("Sex", "Recerational.Athlete", "Birth.Control","PsychDistress"))

    # synth_df = kNNMTD(opt,df,run)
    # synthData_MTD = synth_df.generateData()
    # synthData = model.sample(1000)
    # run['Final Synthetic Data'].log_artifact(synthData)

