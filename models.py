import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from numpy.core.numeric import cross
from torch._C import device
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm
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
from data_transformer import *
from data_sampler import *
import random
import matplotlib.pyplot as plt
from packaging import version
torch.cuda.empty_cache()
KERNEL_TYPE = "multiscale"

#  ---------------  Dataset  ---------------

def getBestK(data):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(data)
            kmeanModel.fit(data)
        
            distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / data.shape[0])
            inertias.append(kmeanModel.inertia_)
        
            mapping1[k] = sum(np.min(cdist(data, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / data.shape[0]
            mapping2[k] = kmeanModel.inertia_
        

    curve = inertias
    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    optimal_K = np.argmax(distToLine)
    return int(optimal_K+1)

class LoadFile():
    """Load dataset."""

    def __init__(self, opt,run):
        """Initializes instance of class.
        Args:
            csv_file (str): Path to the csv file with the data.
        """
        self.run = run
        self.opt = opt
        self.data = pd.read_csv(self.opt.file)
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
        self.train_size = int(0.8 * len(self.data))
        self.test_size = len(self.data) - self.train_size
        self.trainset, self.testset = random_split(self.data, [self.train_size, self.test_size])
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(self.testset, batch_size=self.test_size, shuffle=True)

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
      return self.trainloader,self.testloader


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item),ReLU()]
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


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
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

        seq += [Linear(dim, 2)]
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

def loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
        st = 0
        loss = []
        for column_info in output_info:
            for span_info in column_info:
                if span_info.activation_fn != "softmax":
                    ed = st + span_info.dim
                    std = sigmas[st]
                    loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
                    loss.append(torch.log(std) * x.size()[0])
                    st = ed

                else:
                    ed = st + span_info.dim
                    loss.append(cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                    st = ed

        assert st == recon_x.size()[1]
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


class GenerativeMTD():
    def __init__(self,opt,D_in,run, embedding_dim=128,compress_dims=(256, 256),decompress_dims=(256, 256),l2scale=1e-5,discriminator_lr=2e-4,discriminator_decay=1e-6,discriminator_dim = (256, 256),discriminator_steps=1,generator_lr=2e-4,generator_decay=1e-6,loss_factor=2,batch_size=30,epochs=2,log_frequency=True):
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
        self._epochs = epochs
        self.run = run
        self._log_frequency = log_frequency
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu') 
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.knnmtd_fake = None
        self.criterion = nn.NLLLoss()

        self.run["config/batch_size"] = self._batch_size
        # self.run["config/AE lr"] = self.encoder_lr
        # self.run["config/Discriminator lr"] = self._generator_lr
        self.run["config/AE dim"] = [self.compress_dims, self.embedding_dim,self.decompress_dims]
        self.run["config/discriminator dim"] = self._discriminator_dim
        self.run["config/epoch"] = self.opt.epochs

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
                elif span_info.activation_fn == 'leaky':
                    ed = st + span_info.dim
                    m = nn.LeakyReLU(0.1)
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

    def MMD(self, x, y, kernel):
        """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                    torch.zeros(xx.shape).to(self.device),
                    torch.zeros(xx.shape).to(self.device))
        if kernel == "multiscale":
            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx)**-1
                YY += a**2 * (a**2 + dyy)**-1
                XY += a**2 * (a**2 + dxy)**-1
        if kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        return torch.mean(XX + YY - 2. * XY)
    
    def DeepCoral(self, source, target):
        # d = source.data.shape[1]
        xm = torch.mean(source, 1, keepdim=True)
        xc = torch.matmul(torch.transpose(xm, 0, 1), xm)  # source covariance
        xmt = torch.mean(target, 1, keepdim=True)
        xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)   # target covariance
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))   # frobenius norm between source and target
        return loss



    def fit(self,train_data,discrete_columns=tuple()):
        """Trains the model.
        Args:
            csv_file (str): Absolute path of the dataset used for training.
            n_epochs (int): Number of epochs to train.
        """
        # Use gpu if available
        # dataloader = CreateDatasetLoader(real_df,data, self.opt)
        # train_loader, test_loader, real_data = dataloader.load_data()

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)
        data_dim = self._transformer.output_dimensions
        # self._batch_size = train_data.shape[0]

        generateFake = kNNMTD(self.opt)
        self.knnmtd_fake, knnmtd_fake_numpy = generateFake.generateData(train_data)
        means = self.knnmtd_fake.mean(dim=1, keepdim=True)
        stds = self.knnmtd_fake.std(dim=1, keepdim=True)
        self.knnmtd_fake = (self.knnmtd_fake - means) / stds

        # self.knnmtd_fake = self.knnmtd_fake.to(self.device)
        fake_df = self._transformer.inverse_transform(knnmtd_fake_numpy)
        # self.run["Synth Data"].log(fake_df)
        # fake_df.to_csv("testfile",index=False,float_format='%.3f') #save to file

        fake_dataloader = CreateDatasetLoader(self.knnmtd_fake, self._batch_size,self.opt)
        train_loader, test_loader = fake_dataloader.load_data()
        
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim, self._discriminator_dim).to(self.device)
        print(self.encoder)
        print(self.decoder)
        print(self.discriminator)

        optimizerVAE = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),lr=self._generator_lr,betas=(0.5, 0.9), weight_decay=self._generator_decay)

        optimizerD = Adam(self.discriminator.parameters(), lr=self._discriminator_lr,betas=(0.5, 0.9), weight_decay=self._discriminator_decay)
        

        real = torch.from_numpy(train_data.astype('float32')).to(self.device)
        means = real.mean(dim=1, keepdim=True)
        stds = real.std(dim=1, keepdim=True)
        real = (real - means) / stds
        VAEGLoss = []
        DLoss = []
        mmd_loss = []
        coral_loss = []
        enc_loss = []
        for i in range(self.opt.epochs):
            for ix, data in enumerate(train_loader):
                for step in range(self._discriminator_steps):

                    real_sampled = self.sample_data(real,self._batch_size)
                    real_sampled = real_sampled.to(self.device)
                    fake_knnmtd = data.to(self.device)
                    mu, std, logvar = self.encoder(fake_knnmtd)
                    eps = torch.randn_like(std)
                    emb = eps * std + mu
                    fake, sigmas = self.decoder(emb)
                    fake = self._apply_activate(fake)
                    # fake = fake.detach()
                    optimizerD.zero_grad()
                    y_fake = self.discriminator(fake)
                    y_real = self.discriminator(real_sampled)
                    fake_label = Variable(torch.ones(y_fake.size(0))).to(self.device)
                    real_label = Variable(torch.zeros(y_real.size(0))).to(self.device)
                    y_fake_pred = log_softmax(y_fake, dim=1)
                    y_real_pred = log_softmax(y_real, dim=1)
                    loss_fake_d = self.criterion(y_fake_pred, fake_label.long())
                    loss_real_d = self.criterion(y_real_pred, real_label.long())
                    self.run["loss/Fake Loss"].log(loss_fake_d)
                    self.run["loss/Real Loss"].log(loss_real_d)
                    loss_d = loss_fake_d + loss_real_d
                    loss_d.backward()
                    optimizerD.step()

                
                optimizerVAE.zero_grad()
                real_sampled = self.sample_data(real,self._batch_size)
                real_sampled = real_sampled.to(self.device)
                fake_knnmtd = data.to(self.device)
                mu, std, logvar = self.encoder(fake_knnmtd)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                fake, sigmas = self.decoder(emb)
                fake = self._apply_activate(fake)
                y_fake = self.discriminator(fake)
                fake_label = Variable(torch.ones(y_fake.size(0))).to(self.device)
                y_fake_pred = log_softmax(y_fake, dim=1)
                loss_fake_d = self.criterion(y_fake_pred, fake_label.long())

                KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                mmd = self.MMD(real_sampled, fake, kernel=KERNEL_TYPE)

                loss_g =  KLD + mmd + loss_fake_d
                loss_g.backward()
                optimizerVAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                


                # self.run["loss/G loss"].log(loss_g)
                # self.run["loss/D loss"].log(loss_d)
                # self.run["loss/G MMD"].log(mmd)
                # self.run["loss/G CORAL"].log(coral)
                # self.run["loss/G KLD"].log(KLD)
            # print(KLD)
            # print(mmd)
            # print(loss_g)
            # print(torch.mean(y_fake))
            # print(torch.mean(y_real))
            VAEGLoss.append(loss_g)
            DLoss.append(loss_d)
            self.run["loss/De Loss"].log(loss_g)
            self.run["loss/D Loss"].log(loss_d)
            # self.run["loss/G MMD Loss"].log(mmd_loss)
            # self.run["loss/G CORAL Loss"].log(coral_loss)
            # self.run["loss/G KLD"].log(kld_loss)
            print(f"Epoch {i+1} | Loss De: {loss_g.detach().cpu(): .4f} | "f"Loss D: {loss_d.detach().cpu(): .4f}",flush=True)
        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.arange(self.opt.epochs),VAEGLoss.detach().cpu(),label='Generator Loss')
        plt.plot(np.arange(self.opt.epochs),DLoss.detach().cpu(),label='Discriminator Loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        self.run['loss/loss_plot'].upload(fig)

        # fig = plt.figure(figsize=(15, 15))
        # plt.plot(np.arange(self.opt.epochs),mmd_loss.detach().cpu(),label='MMD Loss')
        # plt.plot(np.arange(self.opt.epochs),coral_loss.detach().cpu(),label='CORAL Loss')
        # plt.xlabel('epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # self.run['loss/MMD+CORAL plot'].upload(fig)
        self.run.stop()

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
    
    def sample(self, samples):
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        self.device = device
        self.decoder.to(self.device)
