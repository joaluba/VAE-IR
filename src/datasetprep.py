import torch
from torch.utils.data import Dataset
import torchaudio
import helpers
import pandas as pd

class DatasetRirs(Dataset):

    def __init__(self,IrInfoFile,sr=8e3,preproc="powspec"):
        self.preproc=preproc
        self.IrInfoFile=IrInfoFile
        self.sr=sr
        self.IrData = pd.read_csv(self.IrInfoFile,delimiter=',')
        self.pad_dur=3 # crop all the irs to this duration [s]

    def __len__(self):
        return len(self.IrData)

    def __getitem__(self,index):

        if self.preproc=="powspec":
            sig, S, minmax=helpers.wav2powspec(self.IrData["filepath"][int(index)], n_fft=1024, hop_length=128, win_length = 256, sample_rate = self.sr, pad_dur=3)
            data_point=S

        elif self.preproc=="wave":
            # load signal
            sig, sr_orig = torchaudio.load(self.IrData["filepath"][int(index)])
            # resample
            sig=torchaudio.transforms.Resample(sr_orig,self.sr)(sig)
            # cut or zero-pad to fixed length
            sig=helpers.cut_or_zeropad(sig,self.pad_dur*self.sr)
            # normalize by standard deviation
            minmax=torch.std(sig) # minmax is the variable name storing standard deviation 
            sig=sig/minmax; minmax=minmax.numpy()
            # store signal as input data point
            data_point=sig.view(1,sig.shape[0],sig.shape[1])
            assert data_point.shape==torch.Size([1,1,int(self.pad_dur*self.sr)]), f"{data_point.shape=}"

        # create label consisting of acoustic params 
        label={
            "isarni":(index-11331)<0,
            "rt": self.IrData["rt"][int(index)],
            "drr": self.IrData["drr"][int(index)],
            "cte": self.IrData["cte"][int(index)],
            "edt": self.IrData["edt"][int(index)],
            "minmax": minmax}
        
        return data_point, label

# check if the dataset definition is correct:
if __name__ == "__main__":

    # instantiate data set
    #INFO_FILE = "/Users/joanna.luberadzka/Documents/VAE-IR/irstats_ARNIandBUT_local.csv"
    INFO_FILE = "/home/ubuntu/joanna/VAE-IR/irstats_ARNIandBUT_datura.csv"
    SAMPLING_RATE=8e3
    # Create dataset object
    dataset_test = DatasetRirs(INFO_FILE,SAMPLING_RATE, "powspec")
    print("Number of data points:" + str(len(dataset_test)))
    print("Dimensions of input data:" + str(dataset_test[20][0].shape))
    print("List of labels:" + str(dataset_test[20][1]))
  

