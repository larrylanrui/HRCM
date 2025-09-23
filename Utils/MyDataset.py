import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        for foldername in os.listdir(root_dir):
            folderpath = os.path.join(root_dir, foldername)
            if os.path.isdir(folderpath):
                for filename in os.listdir(folderpath):
                    filepath = os.path.join(folderpath, filename)
                    image = Image.open(filepath).convert('RGB')
                    self.images.append(image)
                    self.labels.append(int(foldername))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label