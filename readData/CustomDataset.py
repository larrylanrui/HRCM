import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __int__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]