# import torch.utils.data as data
# from sklearn.utils import shuffle
# from PIL import Image
# import os
# import os.path
# import torchvision.transforms.functional as TF
# import random
#
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
# def make_dataset(dir):
#     global gt_name
#     image_pathsFOG = []
#     image_pathsGT = []
#     for file in os.listdir(dir):
#         print(222222222222)
#         print(os.listdir(dir))
#         if not os.path.isdir(file):
#             data_path = os.path.join(dir, file+'/FOG')
#             for root, _, fnames in (os.walk(data_path)):
#                 fnames = shuffle(fnames, n_samples=len(fnames))
#                 for fname in fnames:
#                     fog_path = os.path.join(data_path, fname)
#                     image_pathsFOG.append(fog_path)
#                     # if file == 'SOTS':
#                     #     # gt_name = fname.split('_outdoor')[0]+'.jpg'
#                     #     gt_name = fname
#                     if file == 'denhaze':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#
#                     # if file =='Live_select':
#                     #     gt_name = fname.split('.')[0] + '_our.png'
#                     gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name
#                     image_pathsGT.append(gt_path)
#                     print(fog_path,gt_path)
#
#     return image_pathsGT, image_pathsFOG
#
#
#
#
# def resize(img):
#     width, heigth = img.size
#     new_width = width-(width % 16)
#     new_height = heigth - (heigth % 16)
#     return img.resize((new_width,new_height))
# class ImageFolder(data.Dataset):
#     def __init__(self, root, transform=None, return_paths=True,
#                  loader=default_loader):
#
#         image_pathsGT, image_pathsFOG = make_dataset(root)
#         print(111111111111111111)
#         print(image_pathsGT)
#         print(image_pathsFOG)
#
#         if (len(image_pathsGT) or len(image_pathsFOG)) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#
#         self.image_pathsGT = image_pathsGT
#         self.image_pathsFOG = image_pathsFOG
#         self.transform = transform
#         self.return_paths = return_paths
#         self.loader = loader
#
#     def __getitem__(self, index):
#         pathGT = self.image_pathsGT[index]
#         pathFOG = self.image_pathsFOG[index]
#         imgGT = self.loader(pathGT)
#         imgFOG = self.loader(pathFOG)
#
#
#
#         if self.transform is not None:
#             imgGT = self.transform(imgGT)
#             imgFOG = self.transform(imgFOG)
#
#         if self.return_paths:
#             return imgGT, imgFOG, pathGT, pathFOG
#         else:
#             return imgGT, imgFOG, '', ''
#
#     def __len__(self):
#         return len(self.image_pathsFOG)
#
#
# if __name__ == '__main__':
#     path = "../Datasets/Val"
#     imageFolder = ImageFolder(path)
#
import torch.utils.data as data
from sklearn.utils import shuffle
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as TF
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path)

def make_dataset(dir):
    global gt_name
    image_pathsFOG = []
    image_pathsGT = []

    for file in os.listdir(dir):
        if not os.path.isdir(file):
            data_path = os.path.join(dir, file+'/FOG')
            for root, _, fnames in (os.walk(data_path)):
                fnames = shuffle(fnames, n_samples=len(fnames))
                for fname in fnames:
                    fog_path = os.path.join(data_path, fname)
                    image_pathsFOG.append(fog_path)
                    if file == 'nhhaze':
                        gt_name = fname.split('_')[0] + '_GT.png'
                    elif file == 'SOTS':
                            gt_name = fname

                    # elif file == 'ITS':
                    #     gt_name = fname.split('_')[0] + '.png'
                    gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name


                    image_pathsGT.append(gt_path)

                    # print(fog_path,gt_path)

    return image_pathsGT, image_pathsFOG

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=True,
                 loader=default_loader):

        image_pathsGT, image_pathsFOG= make_dataset(root)

        if (len(image_pathsGT) or len(image_pathsFOG)) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.image_pathsGT = image_pathsGT
        self.image_pathsFOG = image_pathsFOG

        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        pathGT = self.image_pathsGT[index]
        pathFOG = self.image_pathsFOG[index]

        imgGT = self.loader(pathGT)
        imgFOG = self.loader(pathFOG)

        if self.transform is not None:
            imgGT = self.transform(imgGT)
            imgFOG = self.transform(imgFOG)


        if self.return_paths:
            return imgGT, imgFOG, pathGT, pathFOG
        else:
            return imgGT, imgFOG, '', ''

    def __len__(self):
        return len(self.image_pathsFOG)


if __name__ == '__main__':
    path = "../Datasets/Val"
    imageFolder = ImageFolder(path)

