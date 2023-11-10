import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io

class MajajHong2015Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_fpaths_fpath, neural_response_fpath, img_transform=None, response_transform=None):
        self.img_fpaths = np.load(img_fpaths_fpath)
        self.img_transform = img_transform
        self.neural_responses = np.load(neural_response_fpath)
        self.response_transform = response_transform

    def __len__(self): 
        return len(self.img_fpaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fpath = self.img_fpaths[idx]
        img = io.imread(img_fpath)
        response = self.neural_responses[idx]

        if self.img_transform:
            img = self.img_transform(img)
        if self.response_transform:
            response = self.response_transform(response)
            
        return img, response