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

#  ------------------- MODEL DEFINITIONS --------------------

# Definition of 1 residual block:
class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=2, padding=0):
        super().__init__()
        # direct connection
        self.direct_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.direct_bn = nn.BatchNorm1d(out_channels)
        self.direct_relu = nn.PReLU()
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
        x = self.direct_relu(x)
        # add outputs of direct and residual connections
        x += residual
        return x

class sig2ir_decoder(nn.Module):
    def __init__(self, x_len=16000*3, z_len=512, N_layers=10):
        super().__init__() 

        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of residual blocks with increasing channels
        conv_layers = []
        block_channels=1
        for i in range(N_layers):
            conv_layers.append(ResBlock1d(block_channels, block_channels*2, kernel_size=15, stride=stride,padding=padding))
            block_channels*=2
            # compute heigth of the ouput (width=1,depth=block_channels)
            x_len=np.floor((x_len-kernel_size+2*padding)/stride)+1 
        self.conv_layers = nn.Sequential(*conv_layers)

        # adaptive pooling layer to flatten and aggregate information
        self.aggregate = nn.AdaptiveAvgPool1d(1)
        #self.aggregate = nn.Conv1d(block_channels, z_len, kernel_size=int(x_len), stride=1, padding=0)
    
        # final mlp layers
        self.mlp = nn.Linear(z_len, int(z_len/2))


    def forward(self, x):
        # Convolutional residual blocks:
        x = self.conv_layers(x)
        # Aggregate info & flatten:
        x = self.aggregate(x)
        # # Dense layers after aggregation:
        # x = self.mlp(x)
        return x
    

# check if the model definitions are correct
if __name__ == "__main__":
    
    # example input tensor
    x_wave=torch.randn(1,1,24000)

    # instantiate the model
    encoder=sig2ir_decoder(x_len=24000,z_len=512,N_layers=10)
    encoder.to("cpu")
    summary(encoder,input_size = (1, 24000), batch_size = -1, device="cpu")# torch summary expects 2 dim input for MLP
    encoder.eval()
    z=encoder(x_wave)
    print(z.shape)


