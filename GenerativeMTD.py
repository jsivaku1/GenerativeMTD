import numpy as np
from pandas.core.indexes.base import Index
from keras.datasets import mnist
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
from torch._C import device
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from tqdm import tqdm
from geomloss import SamplesLoss
from torchvision import transforms
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader,Dataset
from torch.utils.data import random_split
import torch
import torch.optim as optim
from torch.optim import Adam, SGD
from torch.autograd import Variable, backward
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from torch.nn.functional import cross_entropy, nll_loss, log_softmax
from torch.nn import BatchNorm1d,Parameter, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from ClusterMTD import *
from gvae_data_transformer import DataTransformer
from data_sampler import *
import random
import matplotlib.pyplot as plt
from packaging import version
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from sinkhorn import *
from neptune.new.types import File
torch.cuda.empty_cache()
KERNEL_TYPE = "multiscale"

class LoadFile():
    """Load dataset."""

    def __init__(self, opt,run):
        """Initializes instance of class.
        Args:
            csv_file (str): Path to the csv file with the data.
        """
        self.run = run
        self.opt = opt
        self.data = pd.read_csv(self.opt.dataset)
        self.opt.real_data_dim = [self.data.shape[0], self.data.shape[1]]
        # if(self.opt.choose_best_k):
        #   self.opt.k = getBestK(self.data)
        # self.run["Best K"] = self.opt.k
        self.opt.batch_size = self.opt.k

    def __dim__(self):
        return self.data.shape[1]
        
    # Load dataset
    def load_data(self):
      return self.data, self.opt

class CreateDatasetLoader(Dataset):
    """Load dataset."""

    def __init__(self,data,batch_size,opt):
        """Initializes instance of class.
        Args:
            csv_file (str): Path to the csv file with the data.
        """
        self.opt = opt
        self.batch_size = batch_size
        self.data = data
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu') 
        self.data = self.data.to(self.device)
        # Split into training and test
        self.train_size = int(1.0 * len(self.data))
        self.test_size = len(self.data) - self.train_size
        self.trainset, _ = random_split(self.data, [self.train_size, self.test_size])
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        # self.testloader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True)

    def __len__(self):
        return len(self.data)

    def __dim__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return self.data[idx]

    # Load dataset
    def load_data(self):
      return self.trainloader

    
class VAEncoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(VAEncoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item),LeakyReLU(0.1),Dropout(0.5)]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class VADecoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(VADecoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), LeakyReLU(0.1),Dropout(0.5)]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        dim = input_dim 
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        # self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu') 
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)
        disc_interpolates = self(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - 1) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        return self.seq(input)


class GenerativeMTD():
    def __init__(self,opt,D_in,run, embedding_dim=128,compress_dims=(128,128,128),decompress_dims=(128,128,128),l2scale=1e-5,discriminator_lr=2e-4,discriminator_decay=1e-6,discriminator_dim = (128,128),discriminator_steps=1,generator_lr=2e-4,generator_decay=1e-6,loss_factor=2,batch_size=30,epochs=2,log_frequency=True):
        self.opt = opt
        self.D_in = D_in
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.loss_factor = loss_factor

        self._discriminator_dim = discriminator_dim
        self._discriminator_lr = discriminator_lr
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_decay = discriminator_decay
        self._discriminator_steps = discriminator_steps
        self._batch_size = batch_size
        self._epochs = self.opt.epochs
        self.run = run
        self._log_frequency = log_frequency
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu') 
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.psuedo_fake = None
        self.criterion = nn.NLLLoss()

        self.run["config/batch_size"] = self._batch_size
        self.run["config/AE dim"] = [self.compress_dims, self.embedding_dim,self.decompress_dims]
        self.run["config/discriminator dim"] = self._discriminator_dim
        self.run["config/epoch"] = self._epochs

    def clip_tensors(self, real, fake_transformed):
        for i in np.arange(real.size(1)):
            fake_transformed[:,i] = torch.clamp(fake_transformed[:,i], torch.min(real[:,i]),torch.max(real[:,i]))
        return fake_transformed

    def compute_loss(self, recon_x, x,sigmas,mu_fake,mu_real,std_fake,logvar_fake, factor):
            """Compute the cross entropy loss on the fixed discrete column."""
            loss = []
            conti_col_ix = []
            st = 0
            st_c = 0
            for column_info in self._transformer.output_info_list:
                for span_info in column_info:
                    if len(column_info) != 1 or span_info.activation_fn != "softmax":
                        ed = st + span_info.dim
                        conti_col_ix.append(st)
                        st = ed
                    else:
                        ed = st + span_info.dim
                        ed_c = st_c + span_info.dim
                        tmp = cross_entropy(x[:, st:ed], torch.argmax(recon_x[:, st_c:ed_c], dim=1), reduction='none')
                        loss.append(tmp)
                        st = ed
                        st_c = ed_c
            assert st == recon_x.size()[1]
            if(len(loss) != 0):
                loss = torch.stack(loss, dim=1)
                cross_entropy_loss = (loss * factor).sum() / x.size()[0]
            else:
                cross_entropy_loss = 0
            recon_error = self.MMD(x[:, conti_col_ix], recon_x[:, conti_col_ix])
            # recon_error,_,_ = self.recon_loss(x[:, conti_col_ix], recon_x[:, conti_col_ix])
            div_error,_,_ = self.div_loss(mu_fake, mu_real)
            # div_error = torch.mean(- 0.5 * torch.sum(1 + logvar_fake - mu_fake.pow(2) - logvar_fake.exp(),dim = 1),dim=0)
            return recon_error, div_error, cross_entropy_loss
            # return (recon_error * factor).sum() / x.size()[0], div_error, cross_entropy_loss

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'relu':
                    ed = st + span_info.dim
                    m = nn.ReLU()
                    data_t.append(m(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0
        return torch.cat(data_t, dim=1)

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def MMD(self,x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd
    
    def one_step_epoch(self,data):
        for step in range(self._discriminator_steps):

            # real_sampled = self.sample_data(real,self._batch_size)
            # real_sampled = real_sampled.to(self.device)
            fake_knnmtd = data.to(self.device)
            mu, std, logvar = self.encoder(fake_knnmtd)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            fake, sigmas = self.decoder(emb)
            fakeact = self._apply_activate(fake)
            self.optimizerD.zero_grad()
            y_fake = self.discriminator(fakeact)                    
            loss_fake_d = torch.mean(y_fake)
            y_real = self.discriminator(self.real)
            loss_real_d = torch.mean(y_real)
            pen = self.discriminator.calc_gradient_penalty(self.real, fakeact, self.device)
            pen.backward(retain_graph=True)
            self.loss_d = -(loss_real_d - loss_fake_d)
            self.loss_d.backward()
            self.optimizerD.step()

            self.run["output/Fake Loss"].log(loss_fake_d)
            self.run["output/Real Loss"].log(loss_real_d)

        self.optimizerVAE.zero_grad()

        fake_knnmtd = data.to(self.device)
        mu_real, std_real, logvar_real = self.encoder(self.real)
        mu_fake, std_fake, logvar_fake = self.encoder(fake_knnmtd)
        eps = torch.randn_like(std_fake)
        emb = eps * std_fake + mu_fake
        self.fake, self.sigmas = self.decoder(emb)
        fakeact = self._apply_activate(self.fake)
        y_fake = self.discriminator(fakeact)
        self.loss_fake_d = -torch.mean(y_fake)
        self.recon_error, self.div_error,self.cross_entropy = self.compute_loss(self.fake, self.real,self.sigmas,mu_fake,mu_real,std_fake,logvar_fake, factor=2)
        self.loss_g = self.div_error + self.recon_error + self.cross_entropy + self.loss_fake_d 
        self.loss_g.backward()
        self.optimizerVAE.step()
        
    def fit(self,train_data,discrete_columns=tuple()):
        """Trains the model.
        Args:
            csv_file (str): Absolute path of the dataset used for training.
            n_epochs (int): Number of epochs to train.
        """
        self._batch_size = train_data.shape[0]
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data_transformed = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(train_data_transformed,self._transformer.output_info_list,self._transformer._column_transform_info_list, self._log_frequency)
        data_dim = self._transformer.output_dimensions
        generateFake = kNNMTD(self.opt)
        self.psuedo_fake, psuedo_fake_numpy = generateFake.generateData(train_data_transformed)
        self.run['knnmtd output shape'] = psuedo_fake_numpy.shape
        self.run["real data"].upload(File.as_html(train_data))


        self.fake_data_means = self.psuedo_fake.mean(dim=0)
        self.fake_data_stds = self.psuedo_fake.std(dim=0)
        self.psuedo_fake = (self.psuedo_fake - self.fake_data_means) / self.fake_data_stds
        fake_df = self._transformer.inverse_transform(psuedo_fake_numpy,self.fake_data_means,self.fake_data_stds)
        fake_dataloader = CreateDatasetLoader(self.psuedo_fake, self._batch_size,self.opt)
        train_loader = fake_dataloader.load_data()
        self.real = torch.from_numpy(train_data_transformed.astype('float32')).to(self.device)
        self.real_data_means = self.real.mean(dim=0)
        self.real_data_stds = self.real.std(dim=0)
        self.real = (self.real - self.real_data_means) / self.real_data_stds


        means = torch.stack((self.real_data_means,self.fake_data_means.to(self.device)))
        self.real_fake_means = torch.mean(means,dim=0)
        stds = torch.stack((self.real_data_stds,self.fake_data_stds.to(self.device)))
        self.real_fake_stds = torch.mean(stds,dim=0)


        self.encoder = VAEncoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = VADecoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim, self._discriminator_dim).to(self.device)
        self.optimizerVAE = SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()),lr=self._generator_lr, weight_decay=self._generator_decay, momentum=0.9)
        self.optimizerD = SGD(self.discriminator.parameters(), lr=self._discriminator_lr, weight_decay=self._discriminator_decay, momentum=0.9)

        VAEGLoss = []
        DLoss = []

        self.div_loss = SinkhornDistance(eps=0.01, max_iter=100,device=self.device)

        best_pcd = np.inf
        for i in range(self._epochs):
            for ix, data in enumerate(train_loader):
                self.one_step_epoch(data)

            fake_df = self.transform_to_df(self.fake,self.sigmas)
            self.plot_diagnostics(train_data,fake_df,i)
            curr_pcd = PCD(train_data.copy(),fake_df.copy())

            if(curr_pcd < best_pcd):
                best_pcd = curr_pcd
                torch.save(self.encoder, 'Model/'+ str(self.opt.k) + 'best_encoder' + '_' + self.opt.dataname + '_' + self.opt.model +'.pt')
                torch.save(self.decoder, 'Model/'+ str(self.opt.k) + 'best_decoder'+ '_' + self.opt.dataname + '_' + self.opt.model +'.pt')

            VAEGLoss.append(self.loss_g.item())
            DLoss.append(self.loss_d.item())

            self.run["output/G Loss"].log(self.loss_g)
            self.run["output/D Loss"].log(self.loss_d)
            self.run["output/recon Loss"].log(self.recon_error)
            self.run["output/divergence loss"].log(self.div_error)
            self.run["output/cross entropy loss"].log(self.cross_entropy)
            print(f"Epoch {i+1} | Loss VAE: {self.loss_g.detach().cpu(): .4f} | "f"Loss D: {self.loss_d.detach().cpu(): .4f}",flush=True)
            if(self._epochs % 10 == 0):
                fake_df = self.transform_to_df(self.fake,self.sigmas)
                self.plot_diagnostics(train_data,fake_df,i)
                pcd = PCD(train_data.copy(),fake_df.copy())
                self.run["output/PCD"].log(pcd)
        self.plot_loss(VAEGLoss,DLoss,'VAE+D')

    def sample_data(self, data, n):
        """Sample data from original training data satisfying the sampled conditional vector.
        Returns:
            n rows of matrix data.
        """
        k = n
        # The following code cost 0.2 second
        indice = random.sample(range(k), k)
        indice = torch.tensor(indice)
        sampled_values = data[indice]
        return sampled_values

    def plot_d_losses(self,loss1,loss2,label):
        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.arange(self.opt.epochs),np.array(loss1.detach()),label='D Fake Loss')
        plt.plot(np.arange(self.opt.epochs),np.array(loss2.detach()),label='D Real Loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        self.run['output/'+label].upload(fig)

    def plot_loss(self,loss1,loss2,label):
        fig = plt.figure(figsize=(15, 15))
        plt.plot(loss1,label='Generator Loss')
        plt.plot(loss2,label='Discriminator Loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        self.run['output/'+label].upload(fig)

    def pcd(self, real, fake):
        # real.corr().to_csv('realcorr.csv',index=False)
        # fake.corr().to_csv('fakecorr.csv',index=False)
        pcd = np.linalg.norm((real.corr()-fake.corr()),ord='fro')
        print(pcd)
        return pcd

    def plot_diagnostics(self, real, fake,epoch):
        fig, axs = plt.subplots(5, 4,figsize=(5,5),squeeze=True,frameon=True)
        for ax,col in zip(axs.flat,real.columns):
            sns.kdeplot(real[col],label='Real',ax=ax)
            sns.kdeplot(fake[col],label='Proposed',ax=ax)
            ax.set(xlabel=col, ylabel='')
            ax.get_yaxis().set_label_coords(-0.4,0.5)
            ax.legend([],[], frameon=False)

        fig.delaxes(axs[4,3]) 
        fig.tight_layout()
        axs.flatten()[-2].legend(loc='lower right', bbox_to_anchor=(0.5, -1.5), ncol=3)
        plt.subplots_adjust(top=0.94)
        plt.suptitle(f"Density Plot for epoch {epoch}")
        self.run["output/diagnotics"].upload(fig)

    def sample(self, samples):
        # self.encoder.eval()
        # self.decoder.eval()
        best_encoder = torch.load('Model/'+ str(self.opt.k) + 'best_encoder' + '_' + self.opt.dataname + '_' + self.opt.model +'.pt')
        best_decoder = torch.load('Model/'+ str(self.opt.k) + 'best_decoder'+ '_' + self.opt.dataname + '_' + self.opt.model +'.pt')
        steps = samples // self._batch_size + 1
        data = []
        for _ in range(steps):
            sample_fake = self.sample_data(self.psuedo_fake,self._batch_size)
            fake_knnmtd = sample_fake.to(self.device)
            mu, std, logvar = best_encoder(fake_knnmtd)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            fake, sigmas = best_decoder(emb)
            fake = self._apply_activate(fake)
            data.append(fake.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self._transformer.inverse_transform(data,self.real_fake_means.detach().cpu().numpy(),self.real_fake_stds.detach().cpu().numpy(),sigmas.detach().cpu().numpy())
    
    def transform_to_df(self, fake,sigmas):
        data = []
        data.append(fake.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        return self._transformer.inverse_transform(data,self.real_fake_means.detach().cpu().numpy(),self.real_fake_stds.detach().cpu().numpy(),sigmas.detach().cpu().numpy())

    def set_device(self, device):
        self.device = device
        self.decoder.to(self.device)
