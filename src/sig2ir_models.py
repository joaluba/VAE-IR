import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
# import torchdataset_prep as dsprep
from torchsummary import summary
import argparse
import helpers

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- ENCODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------

# -------- Definition of a convolutional block used in the encoder: --------
class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=2, padding=0):
        super().__init__()
        # direct connection
        self.direct_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.direct_bn = nn.BatchNorm1d(out_channels)
        self.direct_prelu = nn.PReLU()
        # residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # residual connection
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        # direct connection
        x = self.direct_conv(x)
        x = self.direct_bn(x)
        x = self.direct_prelu(x)
        # add outputs of direct and residual connections
        x += residual
        return x
    
# -------- Definition of the encoder: --------
class sig2ir_decoder(nn.Module):
    def __init__(self, x_len=16000*3, z_len=512, N_layers=9):
        super().__init__() 

        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of encoder blocks with increasing channels
        conv_layers = []
        block_channels=1
        for i in range(N_layers):
            conv_layers.append(EncBlock(block_channels, block_channels*2, kernel_size=15, stride=stride,padding=padding))
            block_channels*=2
            # compute heigth of the ouput (width=1,depth=block_channels)
            x_len=np.floor((x_len-kernel_size+2*padding)/stride)+1 

        self.conv_layers = nn.Sequential(*conv_layers)

        # adaptive pooling layer to flatten and aggregate information
        self.aggregate = nn.AdaptiveAvgPool1d(1)
        #self.aggregate = nn.Conv1d(block_channels, z_len, kernel_size=int(x_len), stride=1, padding=0)
    
        # final mlp layers
        self.mlp = nn.Sequential(nn.Linear(block_channels, block_channels),
                                 nn.Linear(block_channels,int(block_channels/2)),
                                 nn.Linear(int(block_channels/2),z_len))
        
    def forward(self, x):
        # Convolutional residual blocks:
        x = self.conv_layers(x)
        # Aggregate info & flatten:
        x = self.aggregate(x)
        x = x.view([x.shape[0],1,-1]) 
        # Dense layers after aggregation:
        x = self.mlp(x)
        return x

# --------------------------------------------------------------------------------------------------
# -------------------------------------------- DECODER: --------------------------------------------
# --------------------------------------------------------------------------------------------------

# -------- Definition of a convolutional block used in the decoder: --------
class GBlock(nn.Module):
    # def __init__(self, in_channels, hidden_channels, z_channels, upsample_factor):
    #     super().__init__()
    #     # parameters of decoder block
    #     self.in_channels = in_channels # this is the size of v 
    #     self.hidden_channels = hidden_channels
    #     self.z_channels = z_channels
    #     self.upsample_factor = upsample_factor
    #     # Decoder block, stage A 
    #     self.A_direct_bn1=nn.BatchNorm1d(in_channels)
    #     self.A_direct_film1=nn.FiLM(in_channels*2,in_channels)
    #     self.A_direct_prelu1=nn.PReLU()
    #     self.A_direct_upsmpl
    #     self.A_direct_conv
    #     self.A_direct_bn2=nn.BatchNorm1d()
    #     self.A_direct_film2
    #     self.A_direct_prelu2
    #     self.A_residual_upsmpl
    #     self.A_residual_conv
    #     # Decoder block, stage B
    #     self.B_direct_bn1=nn.BatchNorm1d()
    #     self.B_direct_film1
    #     self.B_direct_prelu1
    #     self.B_direct_dilconv1
    #     self.B_direct_bn2=nn.BatchNorm1d()
    #     self.B_direct_film2
    #     self.B_direct_prelu2
    #     self.B_direct_dilconv2

    def noise_concat(self, z):
        # concatenate noise and latent variable
        # noise has the same dimension as the latent variable 
        # n= generate noise
        # condition = [n, z]
        # return condition
        return
    
    def forward(self, z, v):
        
        # v - single trainable vector (how to generate this?)
        # z - latent variable
        # inputs = condition

        # outputs= self.A_direct_bn1=nn.BatchNorm1d(self.in_channels)
        # outputs = self.condition_batchnorm1(inputs, z)
        # outputs = self.first_stack(outputs)
        # outputs = self.condition_batchnorm2(outputs, z)
        # outputs = self.second_stack(outputs)

        # residual_outputs = self.residual1(inputs) + outputs

        # outputs = self.condition_batchnorm3(residual_outputs, z)
        # outputs = self.third_stack(outputs)
        # outputs = self.condition_batchnorm4(outputs, z)
        # outputs = self.fourth_stack(outputs)

        # outputs = outputs + residual_outputs
        # return outputs
        return

# -------- Definition of a FiLM layer: --------
class FiLM(nn.Module):
    def __init__(self, zdim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(zdim, n_channels)   
        self.beta = nn.Linear(zdim, n_channels)   

    def forward(self, x, z):
        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        return x

# check if the model definitions are correct
if __name__ == "__main__":
    
    # example input tensor
    FS=48000
    x_len=3*FS
    x_wave=torch.randn(1,1,x_len)

    # instantiate the model
    encoder=sig2ir_decoder(x_len=x_len,z_len=128,N_layers=9)
    encoder.to("cpu")
    summary(encoder,input_size = (1, x_len), batch_size = -1, device="cpu")# torch summary expects 2 dim input for MLP
    encoder.eval()
    z=encoder(x_wave)
    print(f"input shape: {x_wave.shape}")
    print(f"output shape: {z.shape}")

