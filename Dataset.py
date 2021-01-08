import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Dataset(Dataset):

    def __init__(self, root, data, transform = None, train=True):
        self.data = data
        self.root = root
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        img_name = self.data.iloc[idx,0] + '.png'
        img_data = np.array(Image.open(self.root + img_name))
        gt = self.data.iloc[idx,-1]
        gt = np.array(gt)
        if(self.transform):
            img_data = self.transform(img_data)

        if(self.train):
            return {'data':img_data, 'labels':gt}
        else:
            return {'data':img_data}
