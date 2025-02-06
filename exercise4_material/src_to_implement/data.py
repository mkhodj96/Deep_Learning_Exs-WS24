from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        # two different transforms based on whether you are in the training or validation dataset.
        TF = tv.transforms
        self.val_transform = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std),
                                    ])
        self.train_transform = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std)
                                    ])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_row = self.data.iloc[idx]
        img = imread(data_row['filename'], as_gray=True)
        img = gray2rgb(img)  
        
        label = np.array([data_row['crack'], data_row['inactive']], dtype=np.float32)
        
        # Apply the selected transformation
        if self.mode == "val":
            img = self.val_transform(img)
        if self.mode == "train":
            img = self.train_transform(img)
           
        return img, torch.tensor(label, dtype=torch.float32)