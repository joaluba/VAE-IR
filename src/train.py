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

#  ------------------- TRAINING --------------------
# TODO: take params dict as input instead of individual variables

def training(model, dataloader, trainparams, store_outputs):
    # Loss Function, Optimizer and Scheduler
    criterion = trainparams["criterion"]
    optimizer = trainparams["optimizer"]

    outputs=[]
    # Repeat for each epoch
    for epoch in range(trainparams["num_epochs"]):
        running_loss = 0.0

        # Repeat for each batch in the training set
        for i, data in tqdm(enumerate(dataloader)):
            
            # Get the input features and target labels, and put them on the GPU
            x_orig, labels = data[0].to(trainparams["device"]), data[1]

            print(x_orig.shape)
            
            # # Normalize the inputs
            # inputs_m, inputs_s = x_orig.mean(), x_orig.std()
            # x_orig = (x_orig - inputs_m) / inputs_s

            # spectrogram reconstruction (forward pass)
            x_recons, mu, sigma = model(x_orig.to(trainparams["device"]))
            # reconstruction loss 
            recon_loss = criterion(x_orig, x_recons)
            # KL-divergence - measure of similarity between distributions
            kl_div=-torch.sum(1+torch.log(sigma.pow(2))- mu.pow(2)-sigma.pow(2))
            # total loss
            loss=recon_loss+kl_div

            # empty gradient
            optimizer.zero_grad()
            # compute gradients 
            loss.backward()
            # update weights
            optimizer.step()
            # compute loss for the current batch
            running_loss += loss.item()

        # Print stats at the end of the epoch
        num_batches = len(dataloader)
        avg_loss = running_loss / num_batches
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}')
        if store_outputs:
            outputs.append((epoch,x_orig,x_recons,labels))
        
    print('Finished Training')
    if store_outputs:
        return outputs

if __name__ == "__main__":

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "mps")

    # -- Dataset: --
    INFO_FILE = "/Users/joanna.luberadzka/Documents/VAE-IR/irstats_ARNIandBUT.csv"
    SAMPLING_RATE=16e3
    # Create dataset object
    dataset = dsprep.DatasetRirs(INFO_FILE,SAMPLING_RATE)
    # choose a subset for training
    N_IRS=100
    subset_indices = torch.randperm(len(dataset))[:N_IRS]
    
    # -- Model: --
    model=models.VAE_Lin().to(DEVICE)
    
    # -- Training: --
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(subset_indices),num_workers=2)
    LEARNRATE=1e-3
    N_EPOCHS=2
    
    trainparams={
        "num_epochs": N_EPOCHS, 
        "device": DEVICE,
        "learnrate":LEARNRATE,
        "optimizer": torch.optim.Adam(model.parameters(), LEARNRATE),
        "criterion": nn.MSELoss()}

    # training
    train_outputs=training(model, trainloader, trainparams, 1)
    torch.save(model.state_dict(), "/Users/joanna.luberadzka/Documents/VAE-IR/models/trained_model.pth")



