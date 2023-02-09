import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
# import torchdataset_prep as dsprep
import torchsummary 
import argparse

#  ------------------- MODEL DEFINITIONS --------------------

# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)


# class UnFlatten(nn.Module):
#     def forward(self, input, size=1024):
#         return input.view(input.size(0), size, 1, 1)

# class VAE(nn.Module):
#     def __init__(self, x_dim=48000, h_dim=1024, z_dim=32):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv1d(1, 1024, 3, 2,1),
#             nn.ReLU(),
#             nn.Conv1d(1024, 512, 3, 2,1),
#             nn.ReLU(),
#             nn.Conv1d(512, 256, 3, 2,1),
#             nn.ReLU(),
#             nn.Conv2d(256, h_dim, 3,2,1),
#             nn.ReLU(),
#             nn.Flatten(1)
#         )
        
#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)
        
#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
#             nn.Sigmoid(),
#         )
        
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         # return torch.normal(mu, std)
#         esp = torch.randn(*mu.size())
#         z = mu + std * esp
#         return z
    
#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar
        
#     def representation(self, x):
#         return self.bottleneck(self.encoder(x))[0]

#     def forward(self, x):
#         h = self.encoder(x)
#         z, mu, logvar = self.bottleneck(h)
#         z = self.fc3(z)
#         return self.decoder(z), mu, logvar



class Conv1D_VAE(nn.Module):
    def __init__(self,x_len=48000, h_len=256,z_len=24):
        super().__init__()
        self.is_variational=True
        self.conv_layer_1 = nn.Conv1d(1,1024,3,2,1)
        self.norm_layer_1 = nn.BatchNorm1d(1024)
        self.nl_layer_1 = nn.LeakyReLU()

        self.conv_layer_2 = nn.Conv1d(1024, 512,3,2,1)
        self.norm_layer_2 = nn.BatchNorm1d(512)
        self.nl_layer_2 = nn.LeakyReLU()

        self.conv_layer_3 = nn.Conv1d(512, 256,3,2,1)
        self.norm_layer_3 = nn.BatchNorm1d(256)
        self.nl_layer_3 = nn.LeakyReLU()

        self.conv_layer_4 =nn.Conv1d(256, h_len,int(x_len/8),1,0)
        self.norm_layer_4 = nn.BatchNorm1d(h_len)
        self.nl_layer_4 =  nn.LeakyReLU()

        self.fc_mu = nn.Linear(h_len, z_len)
        self.fc_var = nn.Linear(h_len, z_len)

        self.decoder_input = nn.Linear(z_len, h_len)

        self.deconv_layer_1 = nn.ConvTranspose1d(h_len, 256, int(x_len/8),1,0)
        self.deconv_norm_layer_1 = nn.BatchNorm1d(256)
        self.deconv_nl_layer_1 = nn.LeakyReLU()

        self.deconv_layer_2 = nn.ConvTranspose1d(256, 512, 3,2,1, output_padding=1)
        self.deconv_norm_layer_2 = nn.BatchNorm1d(512)
        self.deconv_nl_layer_2 = nn.LeakyReLU()

        self.deconv_layer_3 = nn.ConvTranspose1d(512, 1024, 3,2,1, output_padding=1)
        self.deconv_norm_layer_3 = nn.BatchNorm1d(1024)
        self.deconv_nl_layer_3 = nn.LeakyReLU()

        self.deconv_layer_4 = nn.ConvTranspose1d(1024, 1, 3,2,1,output_padding=1)
        # no activation after last layer - linear 
    
    def myflatten(self,x):
        x_dims=x.shape
        n_batches=x_dims[0]
        unflat_dims=x_dims[1:]
        x=x.view(n_batches, -1)
        return x, unflat_dims
        
    def myunflatten(self,x,unflat_dims):
        dim_final_unflat= tuple([x.size(0)])+tuple(unflat_dims)
        x=x.view(dim_final_unflat)
        return x

    def encode(self, x):

        x = self.conv_layer_1(x)
        x = self.norm_layer_1(x)
        x = self.nl_layer_1(x)

        x = self.conv_layer_2(x)
        x = self.norm_layer_2(x)
        x = self.nl_layer_2(x)

        x = self.conv_layer_3(x)
        x = self.norm_layer_3(x)
        x = self.nl_layer_3(x)

        x = self.conv_layer_4(x)
        x = self.norm_layer_4(x)
        x = self.nl_layer_4(x)
        
        x, self.flatdims=self.myflatten(x)

        mu = self.fc_mu(x)
        var = self.fc_var(x)

        return mu, var

    def reparameterize(self, mu, var):

        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(var/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x):
        x = self.decoder_input(x)
        x = self.myunflatten(x,unflat_dims=self.flatdims)
        x = self.deconv_layer_1(x)
        x = self.deconv_norm_layer_1(x)
        x = self.deconv_nl_layer_1(x)

        x = self.deconv_layer_2(x)
        x = self.deconv_norm_layer_2(x)
        x = self.deconv_nl_layer_2(x)

        x = self.deconv_layer_3(x)
        x = self.deconv_norm_layer_3(x)
        x = self.deconv_nl_layer_3(x)

        x = self.deconv_layer_4(x)
        return x

    def forward(self, x):
         mu, var = self.encode(x)
         z = self.reparameterize(mu, var)
         out = self.decode(z)
         return out, mu, var


class VAE_Lin(nn.Module):
# This is a definition of Variational Autoencoder with MLP layers 
# It expects waveform as input
    def __init__(self, x_len=48000, h_len=256,z_len=24):
        super().__init__() 
        self.is_variational=True
        # ------- Encoder elements: --------
        # q(z|x) - given data, what is the most likely hidden state?
        self.x2h=nn.Sequential(
            nn.Linear(x_len,12000),
            nn.ReLU(),
            nn.Linear(12000,1200),
            nn.ReLU(),
            nn.Linear(1200,h_len)
        )
        self.h2mu=nn.Linear(h_len,z_len)
        self.h2sigma=nn.Linear(h_len,z_len)

        # ------- Decoder elements: --------
        # p(x|z) - given hidden state, what is the likely data for that state?
        self.z2x=nn.Sequential(
            nn.Linear(z_len,1200),
            nn.ReLU(),
            nn.Linear(1200,12000),
            nn.ReLU(),
            nn.Linear(12000,x_len),
            # No activation at the last layer because the standardized waveforms at the input are appnormally distributed (-1,1) 
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


class AutoencoderConv(nn.Module):
    def __init__(self,z_len=24):
        super().__init__()
        self.is_variational=False
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,3,2,1), 
            nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),
            nn.ReLU(),
            nn.Conv2d(32,z_len,(129,47),1,0) 
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(z_len,32,(129,47),1,0), 
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,2,1,output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,2,1,output_padding=(0,1)),
            nn.Sigmoid() # because the values of the (normalized) input are either 0 or 1 
        )

    def forward(self,x):
        x_encoded=self.encoder(x)
        x_decoded=self.decoder(x_encoded)
        return x_decoded, x_encoded



# check if the model definition is correct
if __name__ == "__main__":
    
    # x_wave=torch.randn(1,1,24000)

    # model1=VAE_Lin(x_len=24000,h_len=256,z_len=24)
    # model1.to("cpu")
    # summary(model1,input_size = (1, 24000), batch_size = -1, device="cpu")# torch summary expects 2 dim input for MLP
    # model1.eval()
    # x_recon, mu, sigma=model1(x_wave)
    # print(x_recon.shape)
    # print(mu.shape)
    # print(sigma.shape)

    # x_wave=torch.randn(1,1,24000)
    # model2=Conv1D_VAE(x_len=24000,h_len=256,z_len=24)
    # model2.to("cpu")
    # model2.eval()
    # summary(model2,input_size = (1, 24000), batch_size = -1,device="cpu")# torch summary expects 2 dim input for 1d conv
    # x_recon, mu, sigma=model2(x_wave)
    # print(x_recon.shape)
    # print(mu.shape)
    # print(sigma.shape)

    x_spec=torch.randn(1,513,188)
    model3=AutoencoderConv(z_len=24)
    model3.to("cpu")
    model3.eval()
    t=torchsummary.summary(model3,input_size = (1, 513, 188), batch_size = -1,device="cpu")# torch summary expects 3 dim input for 2d conv
    x_recon, emb =model3(x_spec)
    print(x_recon.shape)
    print(emb.shape)

    