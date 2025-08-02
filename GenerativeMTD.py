# GenerativeMTD.py
# Core model implementation based on the research paper.
# Defines the VAE-GAN architecture and training logic with Sinkhorn and MMD losses.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import json
from collections import defaultdict

from mtd_utils import SinkhornDistance, mmd_rbf, stat_tests

# --- Model Architectures ---

class VAEncoder(nn.Module):
    """Encodes input data into a latent space distribution (mean and log-variance)."""
    def __init__(self, data_dim, c_dims, e_dim):
        super().__init__()
        dim = data_dim
        seq = []
        for item in list(c_dims):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            dim = item
        self.seq = nn.Sequential(*seq)
        self.fc_mu = nn.Linear(dim, e_dim)
        self.fc_logvar = nn.Linear(dim, e_dim)

    def forward(self, x):
        h = self.seq(x)
        return self.fc_mu(h), self.fc_logvar(h)

class VADecoder(nn.Module):
    """Decodes a latent vector back into the data space."""
    def __init__(self, e_dim, d_dims, data_dim, pipeline):
        super().__init__()
        self.pipeline = pipeline
        dim = e_dim
        seq = []
        for item in list(d_dims):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            dim = item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, z):
        """Applies specific activations based on column types."""
        out = self.seq(z)
        final_out = []
        current_idx = 0
        if self.pipeline.numerical_cols:
            num_len = len(self.pipeline.numerical_cols)
            final_out.append(torch.tanh(out[:, current_idx:current_idx+num_len]))
            current_idx += num_len
        if self.pipeline.categorical_cols:
            for num_categories in self.pipeline.cat_lengths:
                cat_part = out[:, current_idx:current_idx+num_categories]
                final_out.append(nn.functional.gumbel_softmax(cat_part, tau=0.2, hard=True))
                current_idx += num_categories
        return torch.cat(final_out, dim=1)

class Critic(nn.Module):
    """Critic network to distinguish real vs. fake data."""
    def __init__(self, data_dim, critic_dims):
        super().__init__()
        dim = data_dim
        seq = []
        for item in list(critic_dims):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.3)]
            dim = item
        seq.append(nn.Linear(dim, 1))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

# --- Main GenerativeMTD Class ---

class GenerativeMTD:
    def __init__(self, real_df_imputed, pipeline, opt, device='cpu'):
        self.real_df_imputed = real_df_imputed
        self.pipeline = pipeline
        self.opt = opt
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.is_trained = False

        self.data_dim = self.pipeline.output_dim
        e_dim = self.opt['embedding_dim']
        c_dims = (e_dim * 2, e_dim * 2)

        self.encoder = VAEncoder(self.data_dim, c_dims, e_dim).to(self.device)
        self.decoder = VADecoder(e_dim, c_dims, self.data_dim, self.pipeline).to(self.device)
        self.critic = Critic(self.data_dim, c_dims).to(self.device)

    def train_vae(self, pseudo_real_df, callback=None):
        """Main training loop for the VAE-GAN."""
        pseudo_trans = self.pipeline.transform(pseudo_real_df)
        X_pseudo = torch.tensor(pseudo_trans, dtype=torch.float32, device=self.device)
        
        real_trans = self.pipeline.transform(self.real_df_imputed)
        X_real = torch.tensor(real_trans, dtype=torch.float32, device=self.device)

        optimizer_G = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.opt['lr'])
        optimizer_D = optim.Adam(self.critic.parameters(), lr=self.opt['lr'])
        
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=self.device)
        batch_size = min(self.opt['batch_size'], X_pseudo.shape[0])
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 10 # Number of epochs to wait for improvement

        for epoch in range(self.opt['epochs']):
            self.encoder.train(); self.decoder.train(); self.critic.train()
            
            perm = torch.randperm(X_pseudo.shape[0])
            epoch_losses = defaultdict(float)
            num_batches_processed = 0

            for i in range(0, X_pseudo.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) == 0: continue
                num_batches_processed += 1

                batch_pseudo = X_pseudo[idx]
                batch_real = X_real[torch.randint(0, len(X_real), (len(idx),))]

                # --- Train Critic ---
                optimizer_D.zero_grad()
                with torch.no_grad():
                    mu, logvar = self.encoder(batch_pseudo)
                    z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                    fake_data = self.decoder(z)
                
                critic_real = self.critic(batch_real).mean()
                critic_fake = self.critic(fake_data).mean()
                loss_d = -(critic_real - critic_fake)
                loss_d.backward()
                optimizer_D.step()

                # --- Train Generator (VAE) ---
                optimizer_G.zero_grad()
                mu_pseudo, logvar_pseudo = self.encoder(batch_pseudo)
                z_pseudo = mu_pseudo + torch.exp(0.5 * logvar_pseudo) * torch.randn_like(mu_pseudo)
                gen_data = self.decoder(z_pseudo)
                
                with torch.no_grad():
                    mu_real, _ = self.encoder(batch_real)
                    z_real = mu_real

                loss_sinkhorn, _, _ = sinkhorn(z_pseudo, z_real)
                
                loss_recon_mmd = torch.tensor(0.0, device=self.device)
                loss_recon_ce = torch.tensor(0.0, device=self.device)
                
                num_cols_len = len(self.pipeline.numerical_cols)
                if self.pipeline.numerical_cols:
                    loss_recon_mmd = mmd_rbf(batch_real[:, :num_cols_len], gen_data[:, :num_cols_len])
                if self.pipeline.categorical_cols:
                    loss_recon_ce = nn.functional.binary_cross_entropy_with_logits(
                        gen_data[:, num_cols_len:], batch_real[:, num_cols_len:]
                    )

                loss_adv = -self.critic(gen_data).mean()
                
                loss_g = loss_sinkhorn + loss_recon_mmd + loss_recon_ce + loss_adv
                loss_g.backward()
                optimizer_G.step()
                
                epoch_losses['Overall Loss'] += loss_g.item()
                epoch_losses['Sinkhorn Loss'] += loss_sinkhorn.item()
                epoch_losses['Reconstruction Loss'] += (loss_recon_mmd + loss_recon_ce).item()
                epoch_losses['Adversarial Loss'] += loss_adv.item()

            # --- Epoch Logging ---
            log_entry = {'status': 'training', 'epoch': epoch + 1, 'total_epochs': self.opt['epochs']}
            num_batches = max(1, num_batches_processed)
            
            with torch.no_grad():
                temp_synth_df = self.sample(len(self.real_df_imputed), is_training=True)
                live_metrics = stat_tests(self.real_df_imputed, temp_synth_df)
                for key, val in live_metrics.items():
                    log_entry[key.upper()] = val if val is not None and not np.isnan(val) else 0.0

            for key, val in epoch_losses.items():
                log_entry[key] = val / num_batches
            
            if callback:
                yield from callback(log_entry)

            # Early stopping check
            current_loss = epoch_losses['Overall Loss'] / num_batches
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.is_trained = True

    def sample(self, n_samples, is_training=False):
        """Generates final synthetic data samples."""
        if not self.is_trained and not is_training:
            raise RuntimeError("Model has not been trained yet.")
        
        if not is_training: self.decoder.eval()
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples // self.opt['batch_size'] + 1):
                z = torch.randn((self.opt['batch_size'], self.opt['embedding_dim']), device=self.device)
                generated = self.decoder(z).cpu().numpy()
                samples.append(np.nan_to_num(generated))
        
        X_fake_transformed = np.concatenate(samples, axis=0)[:n_samples]
        
        return self.pipeline.inverse_transform(X_fake_transformed)