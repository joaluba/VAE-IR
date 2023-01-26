import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
# import torchdataset_prep as dsprep
import argparse

#  ------------------- MODEL DEFINITIONS --------------------

class VAE_Lin(nn.Module):
# This is a definition of Variational Autoencoder with MLP layers 
# It expects waveform as input
    def __init__(self, x_len=48000, h_len=256,z_len=24):
        super().__init__() 
        # ------- Encoder elements: --------
        # q(z|x) - given data, what is the most likely hidden state?
        self.x2h=nn.Sequential(
            nn.Linear(x_len,12000),
            nn.ReLU(),
            nn.Linear(12000,1200),
            nn.ReLU(),
            nn.Linear(1200,600),
            nn.ReLU(),
            nn.Linear(600,h_len)
        )
        self.h2mu=nn.Linear(h_len,z_len)
        self.h2sigma=nn.Linear(h_len,z_len)

        # ------- Decoder elements: --------
        # p(x|z) - given hidden state, what is the likely data for that state?
        self.z2x=nn.Sequential(
            nn.Linear(z_len,600),
            nn.ReLU(),
            nn.Linear(600,1200),
            nn.ReLU(),
            nn.Linear(1200,12000),
            nn.ReLU(),
            nn.Linear(12000,x_len),
            nn.Sigmoid() # because the values of the (normalized) input are either 0 or 1 
        )
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z=mu + sigma * eps
        return z 

    def encode(self,x):             
        # from x_orig to h   
        h=self.x2h(x)
        # from h to mu and sigma 
        mu, sigma = self.h2mu(h), self.h2sigma(h)
        return mu, sigma
    
    def decode(self,z):
        x_recon=self.z2x(z)
        return x_recon

    
    def forward(self,x):
        # 1. Encode
        mu, sigma=self.encode(x)
        # 2. Reparametrize
        z=self.reparameterize(mu,sigma)
        # 3. Decode
        x_recon=self.decode(z)
        return x_recon, mu, sigma 

#  ------------------- MODEL DEFINITION --------------------
class VAE_Lin(nn.Module):
    def __init__(self, x_len=48000, h_len=256,z_len=24):
        super().__init__() 
        # ------- Encoder elements: --------
        # q(z|x) - given data, what is the most likely hidden state?
        self.x2h=nn.Sequential(
            nn.Linear(x_len,12000),
            nn.ReLU(),
            nn.Linear(12000,1200),
            nn.ReLU(),
            nn.Linear(1200,600),
            nn.ReLU(),
            nn.Linear(600,h_len)
        )
        self.h2mu=nn.Linear(h_len,z_len)
        self.h2sigma=nn.Linear(h_len,z_len)

        # ------- Decoder elements: --------
        # p(x|z) - given hidden state, what is the likely data for that state?
        self.z2x=nn.Sequential(
            nn.Linear(z_len,600),
            nn.ReLU(),
            nn.Linear(600,1200),
            nn.ReLU(),
            nn.Linear(1200,12000),
            nn.ReLU(),
            nn.Linear(12000,x_len),
            nn.Sigmoid() # because the values of the (normalized) input are either 0 or 1 
        )
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        z=mu + sigma * eps
        return z 

    def encode(self,x):             
        # from x_orig to h   
        h=self.x2h(x)
        # from h to mu and sigma 
        mu, sigma = self.h2mu(h), self.h2sigma(h)
        return mu, sigma
    
    def decode(self,z):
        x_recon=self.z2x(z)
        return x_recon

    
    def forward(self,x):
        # 1. Encode
        mu, sigma=self.encode(x)
        # 2. Reparametrize
        z=self.reparameterize(mu,sigma)
        # 3. Decode
        x_recon=self.decode(z)
        return x_recon, mu, sigma 

# check if the model definition is correct
if __name__ == "__main__":
    x=torch.randn(1,48000)
    model=VAE_Lin()
    x_recon, mu, sigma=model(x)
    print(x_recon.shape)
    print(mu.shape)
    print(sigma.shape)
    