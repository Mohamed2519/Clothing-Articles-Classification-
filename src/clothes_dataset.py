import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


class ClothesDataset(Dataset):
    def __init__(self, df,label_dict, transform=None):
        self.df = df
        self.transform = transform
        self.label_dict = label_dict
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        img = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['articleType']
        # label_encoded = torch.zeros(11)
        # label_encoded[int(self.label_dict[label])] = 1
        label_encoded = self.label_dict[label]
        if self.transform:
            img = self.transform(img)
        return img, label_encoded


