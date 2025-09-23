# import torch.utils.data as data
# from sklearn.utils import shuffle
# from PIL import Image
# import os
# import os.path
# import torchvision.transforms.functional as TF
# import random
# from scale import Scaling
# from torchvision import transforms
# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
# ]
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#
# def default_loader(path):
#     return Image.open(path)
#
# def make_dataset(dir,opt):
#     global gt_name
#     data_set = opt.TESTING.DATAS
#     image_pathsFOG = []
#     data_path = os.path.join(dir, data_set +'/FOG')
#     # data_path = 'images'
#     for root, _, fnames in (os.walk(data_path)):
#         # fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
#         for fname in fnames:
#             fog_path = os.path.join(data_path, fname)
#             image_pathsFOG.append(fog_path)
#     return image_pathsFOG
# def resize(img):
#     width, heigth = img.size
#     new_width = width-(width % 16)
#     new_height = heigth - (heigth % 16)
#     return img.resize((new_width,new_height))
# class ImageFolder(data.Dataset):
#     def __init__(self, opt, root, transform=None, return_paths=True,
#                  loader=default_loader):
#
#         image_pathsFOG = make_dataset(root,opt)
#
#         if len(image_pathsFOG)==0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#
#         self.image_pathsFOG = image_pathsFOG
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader
#
#
#
#     def __getitem__(self, index):
#         pathFOG = self.image_pathsFOG[index]
#         imgFOG = self.loader(pathFOG)
#
#         old_width, old_height = imgFOG.size
#         # new_width, new_height = img_A.size
#         new_width, new_height = Scaling(old_width, old_height)
#         #
#         #
#         self.transform = transforms.Compose([
#
#             transforms.Lambda(resize),
#             # transforms.Resize(size=(new_height, new_width)),
#             # transforms.Grayscale(num_output_channels=3),
#             # transforms.Resize(size=(672,1024)),
#
#             transforms.ToTensor(),
#             lambda x: x[:3, :, :]
#             # transforms.Normalize((0.5, 0.5, 0.5),
#             #                      (0.5, 0.5, 0.5))
#         ])
#         if self.transform is not None:
#             imgFOG = self.transform(imgFOG)
#
#
#         if self.return_paths:
#             return imgFOG, pathFOG
#         else:
#             return imgFOG, ''
#
#     def __len__(self):
#         return len(self.image_pathsFOG)
#
#
# if __name__ == '__main__':
#     path = "../Datasets/Train"
#     imageFolder = ImageFolder(path)
#
import torch.utils.data as data
from sklearn.utils import shuffle
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as TF
import random
from scale import Scaling
from torchvision import transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path)

def make_dataset(dir,opt):
    global gt_name
    data_set = opt.TESTING.DATAS
    image_pathsFOG = []
    data_path = os.path.join(dir, data_set +'/FOG')
    # data_path = 'images'
    for root, _, fnames in (os.walk(data_path)):
        # fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
        for fname in fnames:
            fog_path = os.path.join(data_path, fname)
            image_pathsFOG.append(fog_path)
    return image_pathsFOG

class ImageFolder(data.Dataset):
    def __init__(self, opt, root, transform=None, return_paths=True,
                 loader=default_loader):

        image_pathsFOG = make_dataset(root,opt)

        if len(image_pathsFOG)==0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.image_pathsFOG = image_pathsFOG
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader


    def __getitem__(self, index):
        pathFOG = self.image_pathsFOG[index]
        imgFOG = self.loader(pathFOG)

        old_width, old_height = imgFOG.size
        # new_width, new_height = img_A.size
        new_width, new_height = Scaling(old_width, old_height)
        #
        #
        self.transform = transforms.Compose([
            # transforms.Resize(size=(new_height, new_width)),
            # transforms.Resize(size=(512,512)),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5),
            #                      (0.5, 0.5, 0.5))
        ])
        if self.transform is not None:
            imgFOG = self.transform(imgFOG)


        if self.return_paths:
            return imgFOG, pathFOG
        else:
            return imgFOG, ''

    def __len__(self):
        return len(self.image_pathsFOG)


if __name__ == '__main__':
    path = "../Datasets/Train"
    imageFolder = ImageFolder(path)

