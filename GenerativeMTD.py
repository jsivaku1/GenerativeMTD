import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import json

from kNNMTD import kNNMTD
from mtd_utils import stat_tests

# --- VAE Model Components ---
class VAEncoder(nn.Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(data_dim, compress_dims), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(compress_dims, compress_dims), nn.LeakyReLU(0.2), nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(compress_dims, embedding_dim)
        self.fc_std = nn.Linear(compress_dims, embedding_dim)

    def forward(self, x):
        h = self.seq(x)
        mu, logvar = self.fc_mu(h), self.fc_std(h)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar

class VADecoder(nn.Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim, decompress_dims), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(decompress_dims, decompress_dims), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(decompress_dims, data_dim)
        )

    def forward(self, z):
        return self.seq(z)

# --- Main GenerativeMTD Class ---
class GenerativeMTD:
    def __init__(self, real_df, opt, device='cpu'):
        self.real_df = real_df 
        self.opt = opt
        self.device = torch.device(device)
        self.encoder = None
        self.decoder = None
        self.data_dim = None

    def generate_pseudo_real_data(self, pipeline):
        """Pipeline Step: Generate pseudo-real data using kNN-MTD."""
        imputed_transformed = pipeline.transform(self.real_df)
        imputed_df = pipeline.inverse_transform(imputed_transformed)
        
        knnmtd = kNNMTD(
            n_obs=self.opt['pseudo_n_obs'], k=self.opt['k'],
            random_state=self.opt['seed'], n_epochs=10
        )
        pseudo_real_df = next(knnmtd.fit_generate(imputed_df), pd.DataFrame())
        if pseudo_real_df.empty:
            raise ValueError("kNN-MTD failed to generate any pseudo-real data.")
        return pseudo_real_df

    def train_vae(self, pseudo_real_df, pipeline, callback=None):
        """Pipeline Step: Transform data and train the VAE."""
        pseudo_real_transformed = pipeline.transform(pseudo_real_df)
        self.data_dim = pseudo_real_transformed.shape[1]

        X = torch.tensor(pseudo_real_transformed, dtype=torch.float32, device=self.device)
        batch_size = min(self.opt['batch_size'], X.shape[0])

        embedding_dim = self.opt['embedding_dim']
        self.encoder = VAEncoder(self.data_dim, embedding_dim * 2, embedding_dim).to(self.device)
        self.decoder = VADecoder(embedding_dim, embedding_dim * 2, self.data_dim).to(self.device)
        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.opt['lr'])

        imputed_transformed = pipeline.transform(self.real_df)
        real_df_imputed = pipeline.inverse_transform(imputed_transformed)

        for epoch in range(self.opt['epochs']):
            self.encoder.train()
            self.decoder.train()
            perm = torch.randperm(X.shape[0])
            
            epoch_loss, epoch_recon_loss, epoch_kld_loss = 0.0, 0.0, 0.0
            for i in range(0, X.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                batch_x = X[idx]

                mu, std, logvar = self.encoder(batch_x)
                eps = torch.randn_like(std)
                emb = mu + eps * std
                recon_x = self.decoder(emb)

                recon_loss = nn.functional.mse_loss(recon_x, batch_x, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kld_loss += kld_loss.item()

            temp_synth_df = self.sample(len(self.real_df), pipeline, is_training=True)
            live_metrics = stat_tests(real_df_imputed, temp_synth_df)

            log_entry = {
                'status': 'training', 'epoch': epoch + 1, 'total_epochs': self.opt['epochs'],
                'loss': epoch_loss / len(X),
                'recon_loss': epoch_recon_loss / len(X),
                'kld_loss': epoch_kld_loss / len(X),
                'pcd': live_metrics.get('pcd'),
                'nndr': live_metrics.get('nndr'),
                'dcr': live_metrics.get('dcr')
            }
            if callback:
                yield f"data: {json.dumps(log_entry)}\n\n"

    def sample(self, n, pipeline, is_training=False):
        """Pipeline Step: Sample from VAE and inverse-transform the data."""
        if not self.decoder or not self.encoder:
            raise RuntimeError("Model has not been trained. Call train_vae() first.")
        
        if not is_training: self.decoder.eval()

        samples_transformed = []
        steps = n // self.opt['batch_size'] + 1
        with torch.no_grad():
            for _ in range(steps):
                z = torch.randn((self.opt['batch_size'], self.opt['embedding_dim']), device=self.device)
                fake_transformed = self.decoder(z)
                samples_transformed.append(fake_transformed.cpu().numpy())
        
        X_fake_transformed = np.concatenate(samples_transformed, axis=0)[:n]
        X_fake_transformed = np.nan_to_num(X_fake_transformed)
        
        return pipeline.inverse_transform(X_fake_transformed)