import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .image_folder_test import ImageFolder


class PairedData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        imgFOG, pathFOG = next(self.data_loader_iter)

        return {'FOG_images': imgFOG, 'FOG_paths': pathFOG}


class UnalignedDataLoader_test(object):
    def __init__(self, params):
        transform = transforms.Compose([
            # transforms.Resize((1024,672)),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5),
            #                      (0.5, 0.5, 0.5))
        ])
        #
        dataset_train = torch.utils.data.DataLoader(
            ImageFolder(opt=params, root=params.TRAINING.VAL_DIR + '/' + 'Test', transform=transform,),
            batch_size=1,num_workers=8, shuffle=False, drop_last=False, pin_memory=True)

        self.dataset_train = dataset_train
        self.paired_data = PairedData(self.dataset_train)

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_train)
