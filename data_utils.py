import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature
from utils import pad
from dataio import pad as pad_cnsl
			
class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, cut=66800, format='.flac'):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=cut
        self.format=format
        print('train/cut:', cut)
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        if 'normal' in utt_id:
            X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000) # 
        else:
            X, fs = librosa.load(self.base_dir+utt_id+self.format, sr=16000)
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target

class Dataset_train_cnsl(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, cut=66800, format='.flac', random_start=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=cut
        self.format=format
        self.random_start=random_start
        print('train/cut:', cut)
        print('train/random_start:', random_start)
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        if 'normal' in utt_id:
            X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000) # 
        else:
            X, fs = librosa.load(self.base_dir+utt_id+self.format, sr=16000)
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad= pad_cnsl(Y, padding_type='repeat', max_len=self.cut, random_start=self.random_start)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track, cut=66800, format='.flac'):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut # take ~4 sec audio 
        self.track = track
        self.format=format
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+self.format, sr=16000)
        if self.cut < 0:
            X_pad = X
        else:
            X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id 
