import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import datasetprep as dsprep
import models
import argparse
import pandas as pd
import getpass
import helpers

#  ------------------- TRAINING --------------------

def training(model, dataloader, valloader, trainparams, store_outputs):
    # Loss Function, Optimizer and Scheduler
    criterion = trainparams["criterion"]
    optimizer = trainparams["optimizer"]
    num_epochs=trainparams["num_epochs"]
    device=trainparams["device"]

    outputs_evol=[]
    loss_evol=[]
    
    for epoch in range(num_epochs):
        

        # Training loop for this epoch: 
        model.train()
        train_loss=0
        for i, data in tqdm(enumerate(dataloader)):
            
            # Get the input features and target labels, and put them on the GPU
            x_orig, labels = data[0].to(device), data[1]
            
            # Standardize the inputs
            inputs_m, inputs_s = x_orig.mean(), x_orig.std()
            x_orig = (x_orig - inputs_m) / inputs_s

            if model.is_variational:
                # reconstruction (forward pass)
                x_recons, mu, sigma = model(x_orig.to(device))
                # reconstruction loss 
                recon_loss = criterion(x_orig, x_recons)
                # KL-divergence - measure of similarity between distributions
                kl_div=-torch.sum(1+torch.log(sigma.pow(2))- mu.pow(2)-sigma.pow(2))
                # total loss
                loss=recon_loss+kl_div
            else:
                # reconstruction (forward pass)
                x_recons, mu= model(x_orig.to(device))
                # reconstruction loss 
                recon_loss = criterion(x_orig, x_recons)
                # total loss
                loss=recon_loss

            # empty gradient
            optimizer.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer.step()
            # compute loss for the current batch
            train_loss += loss.item()

        # If needed - store last batch of the epoch
        if store_outputs:
            outputs_evol.append((epoch,x_orig.cpu().detach().numpy(),x_recons.cpu().detach().numpy(),labels))

        # Validation loop for this epoch: 
        model.eval() 
        with torch.no_grad():
            val_loss=0
            for j, data in tqdm(enumerate(valloader)):

                # Get the input features and target labels, and put them on the GPU
                x_orig, labels = data[0].to(device), data[1]
                # Standardize the inputs
                inputs_m, inputs_s = x_orig.mean(), x_orig.std()
                x_orig = (x_orig - inputs_m) / inputs_s
                
                if model.is_variational:
                    # reconstruction (forward pass)
                    x_recons, mu, sigma = model(x_orig.to(device))
                    # reconstruction loss 
                    recon_loss = criterion(x_orig, x_recons)
                    # KL-divergence - measure of similarity between distributions
                    kl_div=-torch.sum(1+torch.log(sigma.pow(2))- mu.pow(2)-sigma.pow(2))
                    # total loss
                    loss=recon_loss+kl_div
                else:
                    # reconstruction (forward pass)
                    x_recons, mu= model(x_orig.to(device))
                    # reconstruction loss 
                    recon_loss = criterion(x_orig, x_recons)
                    # total loss
                    loss=recon_loss
          
                # compute loss for the current batch
                val_loss += loss.item()
        

        # If needed - store losses
        if store_outputs:
            loss_evol.append((train_loss,val_loss))
        

        # Print stats at the end of the epoch
        num_samples_train=len(dataloader.sampler)
        num_samples_val=len(valloader.sampler)
        avg_train_loss = train_loss / num_samples_train
        avg_val_loss = val_loss / num_samples_val
        print(f'Epoch: {epoch}, Train. Loss: {avg_train_loss:.5f}, Val. Loss: {avg_val_loss:.5f}')
        # Store 


    print('Finished Training')
    if store_outputs:
        return outputs_evol, loss_evol

# IR loss from some paper
class IRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,h1,h2):
        numer=torch.sum(h1*h2,axis=1,keepdim=True)
        denom=torch.norm(h1, p='fro',dim=1,keepdim=True)*torch.norm(h2, p='fro',dim=1,keepdim=True)
        loss=torch.sum(1-torch.pow(numer/denom,2))
        return loss 

if __name__ == "__main__":

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps")

    # -- Data: --
    if getpass.getuser()=="joanna.luberadzka":
        projectdir= "/Users/joanna.luberadzka/Documents/VAE-IR/"
        INFO_FILE = projectdir + "irstats_ARNIandBUT_datura.csv"
    elif getpass.getuser()=="ubuntu":
        projectdir="/home/ubuntu/joanna/VAE-IR/"
        INFO_FILE = projectdir+"irstats_ARNIandBUT_datura.csv"


    # Create dataset object
    SAMPLING_RATE=8e3
    dataset = dsprep.DatasetRirs(INFO_FILE,SAMPLING_RATE,"powspec")

    # split dataset into training set, test set and validation set
    N_train = round(len(dataset) * 0.1)
    N_rest = len(dataset) - N_train
    trainset, restset = random_split(dataset, [N_train, N_rest])
    N_test = round(len(restset) * 0.5)
    N_val = len(restset) - N_test
    testset, valset = random_split(restset, [N_test, N_val])

    # create dataloaders
    BATCH_SIZE=16
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6,pin_memory=True)
    
    # -- Model: --
    model=models.AutoencoderConv(z_len=24).to(DEVICE)
    # model=models.VAE_Lin(x_len=24000).to(DEVICE)
    # model=models.Conv1D_VAE(x_len=24000,h_len=256,z_len=24).to(DEVICE)
    
    # -- Training: --
    LEARNRATE=1e-3
    N_EPOCHS=10
    trainparams={
        "num_epochs": N_EPOCHS, 
        "device": DEVICE,
        "learnrate":LEARNRATE,
        "optimizer": torch.optim.Adam(model.parameters(), LEARNRATE),
        "criterion": nn.MSELoss()}

    # training
    outputs_evol, loss_evol=training(model, trainloader, valloader, trainparams, 1)
    torch.save(model.state_dict(), projectdir + "models/trained_model.pth")





