import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .image_folder import  ImageFolder
import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os


class PairedData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        imgGT, imgFOG, pathGT, pathFOG = next(self.data_loader_iter)
        return {'GT_images': imgGT, 'GT_paths': pathGT,
                'FOG_images': imgFOG, 'FOG_paths': pathFOG}


class UnalignedDataLoader_train(object):
    def __init__(self, params):
        transform = transforms.Compose([
            # transforms.Resize(size=(512,512)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ToTensor(),
            lambda x:x[:3,:,:]
            # transforms.Normalize((0.5, 0.5, 0.5),
            #                      (0.5, 0.5, 0.5))
        ])

        dataset_train = torch.utils.data.DataLoader(
            ImageFolder(opt=params, root=params.TRAINING.TRAIN_DIR + '/' + 'Train', transform=transform,),
            batch_size=params.OPTIM.BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False, pin_memory=False)

        self.dataset_train = dataset_train
        self.paired_data = PairedData(self.dataset_train)

    def load_data(self):
        return self.paired_data
    def __len__(self):
        return len(self.dataset_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../datasets/trainData', type=str)
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--height', default=256, type=int)
    parser.add_argument('--load_size', default=142, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--phase', default='train', type=str)
    params = parser.parse_args()

    warnings.filterwarnings("error", category=UserWarning)
    unalignedDataLoader = UnalignedDataLoader_train(params)
    dataset = unalignedDataLoader.load_data()
    i = 0
    for _, u in enumerate(dataset):

        i += 1
        # if u['FOG_images'].size(1)==4 or u['GT_images'].size(1)==4 or u['FOG_images'].shape != u['GT_images'].shape:
        #     print(u['FOG_paths'])
        #     os.remove(u['FOG_paths'][0])

        img_A = torchvision.utils.make_grid(u['FOG_images']).numpy()
        img_B = torchvision.utils.make_grid(u['GT_images']).numpy()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))
        axes[0].imshow(np.transpose(img_A*0.5+0.5, (1, 2, 0)))
        axes[1].imshow(np.transpose(img_B*0.5+0.5, (1, 2, 0)))
        plt.show()