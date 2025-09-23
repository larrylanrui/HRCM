import torch
from sympy.physics.vector.printing import params
from torch.utils.data import DataLoader
from torchvision import transforms
from .image_folder_val import ImageFolder


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


class UnalignedDataLoader_val(object):
    def __init__(self, params=None):
        transform = transforms.Compose([
            # transforms.Resize((512,512)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=(-45, 45)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5),
            #                      (0.5, 0.5, 0.5))
        ])
        # print(33333333333333333333)
        # print(params.TRAINING.VAL_DIR)
        dataset_train = torch.utils.data.DataLoader(
            ImageFolder(root=params.TRAINING.VAL_DIR + '/' + 'Val', transform=transform, ),
            batch_size=1, num_workers=8, shuffle=False, drop_last=False, pin_memory=False)
        self.dataset_train = dataset_train
        self.paired_data = PairedData(self.dataset_train)

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_train)

if __name__ == '__main__':
    val_dataset = UnalignedDataLoader_val()
    val_loader = val_dataset.load_data()
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val['HR_images'].cuda()
        input_ = data_val['X4_images'].cuda()
        print(input_.shape)
        print(target.shape)
        print("###################################")