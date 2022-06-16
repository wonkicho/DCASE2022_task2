import torch
from torch.utils.data import Dataset
import torchaudio
import joblib
import librosa
import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


class Wav_Mel_ID_Dataset(Dataset):
    def __init__(self, root_path , machine_type, csv_list, unique_sec, sr, states,
                 win_length, hop_length, transform=None):
        
        self.root_path = root_path
        self.machine_type = machine_type
        self.csv_path_list = csv_list
        self.transform = transform
        self.sr = sr
        self.states = states
        self.win_len = win_length
        self.hop_len = hop_length

        
        self.sec = []
        self.unique_sec = unique_sec
        self.attr = []
        self.labels = []
        self.unique_attr = []
        self.file_path_list = []
        for c in self.csv_path_list:
            csv_path = os.path.join(self.root_path,self.machine_type, c)
            
            df = pd.read_csv(csv_path)
            
            for p, v in zip(df["file_name"].values, df["d1v"].values):
                if len(p.split('/')) == 1 and (machine_type == "ToyCar" or machine_type == "ToyTrain"):
                    p = os.path.join(machine_type, "train", p)

                
                if p.split('/')[1] == self.states:
                    fp = os.path.join(self.root_path, p)
                    self.file_path_list.append(fp)
                    self.attr.append(v)

                
                    
                s = re.findall('section_[0-9][0-9]', p)[0]
                self.sec.append(s)
                # else:
                #     print(p)
                #     fp = os.path.join(self.root_path, p)
                #     self.file_path_list.append(fp)
                #     self.attr.append(v)
            
            for v in df["d1v"].unique():
                self.unique_attr.append(v)
        
        

    def __getitem__(self, item):
        file_path = self.file_path_list[item]
        #label = self.unique_attr.index(self.attr[item])
        #label = self.unique_sec.index(self.sec[item])
        label = []
        for i in range(len(self.unique_sec)):
            if i == self.unique_sec.index(self.sec[item]):
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)


        #label = self.labels[item]
        (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

        x = x[:self.sr * 10]  # (1, audio_length)
        x_wav = torch.from_numpy(x)
        x_mel = self.transform(x_wav).unsqueeze(0)
        
        # print(x.shape)

        return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_path_list)


class WavMelClassifierDataset:
    def __init__(self, root_path, machine_type, csv_list, unique_sec, sr, states):
        self.root_path = root_path
        self.machine_type = machine_type
        self.unique_sec = unique_sec
        self.csv_list = csv_list
        self.sr = sr
        self.states = states
        

    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        dataset = Wav_Mel_ID_Dataset(self.root_path,
                                     self.machine_type,
                                     self.csv_list,
                                     self.unique_sec,
                                     self.sr,
                                     self.states,
                                     win_length,
                                     hop_length,
                                     transform=Generator(
                                         self.sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         win_length=win_length,
                                         hop_length=hop_length,
                                         power=power,
                                     ))

        return dataset