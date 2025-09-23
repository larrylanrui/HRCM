# import torch.utils.data as data
# from sklearn.utils import shuffle
# from PIL import Image
# import os
# import os.path
# import torchvision.transforms.functional as TF
# import random
# import numpy as np
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
#         if not os.path.isdir(file):
#             data_path = os.path.join(dir, file+'/FOG')
#             for root, _, fnames in (os.walk(data_path)):
#                 fnames = shuffle(fnames, n_samples=len(fnames))
#                 for fname in fnames:
#
#                     fog_path = os.path.join(data_path, fname)
#                     image_pathsFOG.append(fog_path)
#                     if file == 'xxxxx':
#                         gt_name = fname.split('-')[0]+'-0.jpg'
#                     elif file == 'HazeRD_R':
#                         gt_name = fname.split('-')[0] + '.jpg'
#                     elif file=='NTIRE':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#                     elif file=='SOTS_C':
#                         gt_name = fname
#                     elif file=='OTS':
#                         gt_name = fname.split('_')[0] + '.png'
#                     elif file=='ITS_C':
#                         gt_name = fname
#                     elif file=='FRIDA':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#                     elif file == 'I-HAZE_R':
#                         gt_name = fname.split('hazy')[0] + 'GT.jpg'
#                     elif file == 'O-HAZE_R':
#                         # gt_name = fname.split('hazy')[0] + 'GT.jpg'
#                         gt_name = fname.split('_')[0] + '.jpg'
#                     elif file == 'O-HAZE_R_5':
#                         gt_name = fname.split('hazy')[0] + 'GT.jpg'
#                     # elif file == 'NH-HAZE':
#                     #     gt_name = fname.split('hazy')[0] + 'GT.png'
#                     elif file == 'NH-HAZE':
#                         gt_name = fname.replace('hazy', 'GT')
#                     elif file == 'Dense-Haze':
#                         gt_name = fname.split('hazy')[0] + 'GT.png'
#                     elif file == 'MRFID':
#                         gt_name = fname.split('-')[0] + '.jpg'
#                     gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name
#                     image_pathsGT.append(gt_path)
#                     # print(fog_path,gt_path)
#     return image_pathsGT, image_pathsFOG
#
# class ImageFolder(data.Dataset):
#     def __init__(self, opt, root, transform=None, return_paths=True,
#                  loader=default_loader):
#
#         image_pathsGT, image_pathsFOG = make_dataset(root)
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
#         self.ps = opt.TRAINING.TRAIN_PS
#         # self.ps = 512
#
#
#     def __getitem__(self, index):
#         pathGT = self.image_pathsGT[index]
#         pathFOG = self.image_pathsFOG[index]
#         imgGT = self.loader(pathGT)
#         imgFOG = self.loader(pathFOG)
#
#         ps = self.ps
#         FOG_w, FOG_h = imgFOG.size
#
#         padw = ps-FOG_w if FOG_w<ps else 0
#         padh = ps-FOG_h if FOG_h<ps else 0
#
#         # Reflect Pad in case image is smaller than patch_size
#         if padw!=0 or padh!=0:
#             imgFOG = TF.pad(imgFOG, (0,0,padw,padh), padding_mode='reflect')
#             imgGT = TF.pad(imgGT, (0,0,padw,padh), padding_mode='reflect')
#
#         if self.transform is not None:
#             seed = np.random.randint(2147483647)  # make a seed with numpy generator
#             random.seed(seed)  # apply this seed to img tranfsorms
#             imgGT = self.transform(imgGT)
#             random.seed(seed)  # apply this seed to img tranfsorms
#             imgFOG = self.transform(imgFOG)
#
#         hh, ww = imgFOG.shape[1], imgFOG.shape[2]
#
#         rr     = random.randint(0, hh-ps)
#         cc     = random.randint(0, ww-ps)
#
#         # Crop patch
#         imgFOG = imgFOG[:, rr:rr+ps, cc:cc+ps]
#         imgGT = imgGT[:, rr:rr+ps, cc:cc+ps]
#         # print(imgFOG.shape,imgGT.shape,pathFOG)
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
#     path = "../Datasets/Train"
#     imageFolder = ImageFolder(path)
#
import torch
import torch.utils.data as data
from sklearn.utils import shuffle
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as TF
import random
import numpy as np


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
    # image_pathsDep = []
    for file in os.listdir(dir):
        if not os.path.isdir(file):
            data_path = os.path.join(dir, file+'/FOG')
            for root, _, fnames in (os.walk(data_path)):
                fnames = shuffle(fnames, n_samples=len(fnames))
                for fname in fnames:
                    fog_path = os.path.join(data_path, fname)
                    image_pathsFOG.append(fog_path)
                    if file == 'MRFID':
                        gt_name = fname.split('-')[0]+'-0.jpg'
                    elif file == 'HazeRD_R':
                        gt_name = fname.split('-')[0] + '.jpg'
                        # dep_name = fname.split('.')[0] + '_disp.jpeg'
                    elif file=='NTIRE':
                        gt_name = fname.split('_')[0] + '_GT.png'
                    elif file=='OTS':
                        gt_name = fname.split('_')[0] + '.jpg'
                        # dep_name = fname.split('.jpg')[0] + '_disp.jpeg'
                    elif file=='ITS':
                        gt_name = fname.split('_')[0] + '_.png'
                        # dep_name = fname.split('.png')[0] + '_disp.jpeg'
                    elif file=='FRIDA':
                        gt_name = fname.split('_')[0] + '_GT.png'
                        # dep_name = fname.split('.png')[0] + '_disp.jpeg'
                    elif file=='I-HAZE_R':
                        gt_name = fname.split('hazy')[0] + 'GT.jpg'
                        # dep_name = fname.split('.jpg')[0] + '_disp.jpeg'
                    elif file == 'O-HAZE':
                        gt_name = fname.split('hazy')[0] + 'GT.jpg'
                    elif file == 'NH-HAZE':
                        gt_name = fname.split('_')[0] + '_GT.png'
                    elif file == 'denhaze':
                        gt_name = fname.split('_')[0] + '_GT.png'

                    gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name
                    # dep_path = fog_path.split('FOG/')[0] + 'Dep/' + dep_name
                    image_pathsGT.append(gt_path)
                    # image_pathsDep.append(dep_path)
                    # print(fog_path,dep_path)
    return image_pathsGT, image_pathsFOG

#
# def make_dataset(dir):
#     global gt_name
#     image_pathsFOG = []
#     image_pathsGT = []
#     image_pathsDep = []
#     for file in os.listdir(dir):
#         if not os.path.isdir(file):
#             data_path = os.path.join(dir, file+'/FOG')
#             for root, _, fnames in (os.walk(data_path)):
#                 fnames = shuffle(fnames, n_samples=len(fnames))
#                 for fname in fnames:
#                     fog_path = os.path.join(data_path, fname)
#                     image_pathsFOG.append(fog_path)
#                     if file == 'MRFID':
#                         gt_name = fname.split('-')[0]+'-0.jpg'
#                     elif file == 'HazeRD_R':
#                         gt_name = fname.split('-')[0] + '.jpg'
#                         dep_name = fname.split('.')[0] + '_disp.jpeg'
#                     elif file=='NTIRE':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#                     elif file=='OTS':
#                         gt_name = fname.split('_')[0] + '.jpg'
#                         dep_name = fname.split('.jpg')[0] + '_disp.jpeg'
#                     elif file=='ITS':
#                         gt_name = fname.split('_')[0] + '_.png'
#                         dep_name = fname.split('.png')[0] + '_disp.jpeg'
#                     elif file=='FRIDA':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#                         dep_name = fname.split('.png')[0] + '_disp.jpeg'
#                     elif file=='I-HAZE_R':
#                         gt_name = fname.split('hazy')[0] + 'GT.jpg'
#                         dep_name = fname.split('.jpg')[0] + '_disp.jpeg'
#                     elif file == 'O-HAZE_R':
#                         gt_name = fname.split('hazy')[0] + 'GT.jpg'
#                         dep_name = fname.split('.jpg')[0] + '_disp.jpeg'
#                     gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name
#                     dep_path = fog_path.split('FOG/')[0] + 'Dep/' + dep_name
#                     image_pathsGT.append(gt_path)
#                     image_pathsDep.append(dep_path)
#                     # print(fog_path,dep_path)
#     return image_pathsGT, image_pathsFOG, image_pathsDep


# def make_dataset(dir):
#     global gt_name
#     image_pathsFOG = []
#     image_pathsGT = []
#     for file in os.listdir(dir):
#         if not os.path.isdir(file):
#             data_path = os.path.join(dir, file+'/FOG')
#             for root, _, fnames in (os.walk(data_path)):
#                 fnames = shuffle(fnames, n_samples=len(fnames))
#                 for fname in fnames:
#                     fog_path = os.path.join(data_path, fname)
#                     image_pathsFOG.append(fog_path)
#                     if file == 'MRFID':
#                         gt_name = fname.split('-')[0]+'-0.jpg'
#                     elif file == 'HazeRD':
#                         gt_name = fname.split('-')[0] + '_GT.png'
#                     elif file=='NTIRE':
#                         gt_name = fname.split('_')[0] + '_GT.png'
#                     elif file=='OTS':
#                         gt_name = fname.split('_')[0] + '.jpg'
#                     elif file=='ITS':
#                         gt_name = fname.split('_')[0] + '.png'
#                     gt_path = fog_path.split('FOG/')[0] + 'GT/' + gt_name
#                     image_pathsGT.append(gt_path)
#                     # print(fog_path,gt_path)
#     return image_pathsGT, image_pathsFOG

class ImageFolder(data.Dataset):
    def __init__(self, opt, root, transform=None, return_paths=True,
                 loader=default_loader):

        image_pathsGT, image_pathsFOG= make_dataset(root)
        # print(image_pathsDep)

        if (len(image_pathsGT) or len(image_pathsFOG)) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.image_pathsGT = image_pathsGT
        self.image_pathsFOG = image_pathsFOG
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

        self.ps = opt.TRAINING.TRAIN_PS
        # self.ps = 512

    # def __getitem__(self, index):
    #     pathGT = self.image_pathsGT[index]
    #     pathFOG = self.image_pathsFOG[index]
    #     pathDep = self.image_pathsDep[index]
    #
    #     # 确保所有路径有效
    #     if not os.path.exists(pathGT) or not os.path.exists(pathFOG):
    #         print(f"警告：图像路径不存在 - GT: {pathGT}, FOG: {pathFOG}")
    #         # 返回占位符数据
    #         return self._get_placeholder_data()
    #
    #     # 尝试加载图像，处理可能的异常
    #     try:
    #         imgGT = self.loader(pathGT)
    #         imgFOG = self.loader(pathFOG)
    #     except Exception as e:
    #         print(f"错误：加载图像失败 - {e}, GT: {pathGT}, FOG: {pathFOG}")
    #         return self._get_placeholder_data()
    #
    #     # 处理深度图
    #     if pathDep is not None and os.path.exists(pathDep):
    #         try:
    #             imgDep = self.loader(pathDep)
    #         except Exception as e:
    #             print(f"警告：加载深度图失败 - {e}, Dep: {pathDep}")
    #             imgDep = Image.new('RGB', (self.ps, self.ps), color='black')
    #     else:
    #         imgDep = Image.new('RGB', (self.ps, self.ps), color='black')
    #
    #     ps = self.ps
    #     FOG_w, FOG_h = imgFOG.size
    #
    #     padw = ps - FOG_w if FOG_w < ps else 0
    #     padh = ps - FOG_h if FOG_h < ps else 0
    #
    #     # Reflect Pad in case image is smaller than patch_size
    #     if padw != 0 or padh != 0:
    #         imgFOG = TF.pad(imgFOG, (0, 0, padw, padh), padding_mode='reflect')
    #         imgGT = TF.pad(imgGT, (0, 0, padw, padh), padding_mode='reflect')
    #         imgDep = TF.pad(imgDep, (0, 0, padw, padh), padding_mode='reflect')
    #
    #     if self.transform is not None:
    #         seed = np.random.randint(2147483647)
    #         random.seed(seed)
    #         imgGT = self.transform(imgGT)
    #         random.seed(seed)
    #         imgFOG = self.transform(imgFOG)
    #         random.seed(seed)
    #         imgDep = self.transform(imgDep)
    #
    #     hh, ww = imgFOG.shape[1], imgFOG.shape[2]
    #
    #     rr = random.randint(0, hh - ps)
    #     cc = random.randint(0, ww - ps)
    #
    #     # Crop patch
    #     imgFOG = imgFOG[:, rr:rr + ps, cc:cc + ps]
    #     imgGT = imgGT[:, rr:rr + ps, cc:cc + ps]
    #     imgDep = imgDep[:, rr:rr + ps, cc:cc + ps]
    #
    #     if self.return_paths:
    #         return imgGT, imgFOG, imgDep, pathGT, pathFOG, pathDep
    #     else:
    #         return imgGT, imgFOG, imgDep, '', ''
    #
    # def _get_placeholder_data(self):
    #     """生成占位符数据，确保返回类型正确"""
    #     # 创建全黑的PIL Image，然后转换为Tensor
    #     img = Image.new('RGB', (self.ps, self.ps), color='black')
    #     if self.transform is not None:
    #         img_tensor = self.transform(img)
    #     else:
    #         img_tensor = torch.zeros(3, self.ps, self.ps)
    #
    #     # 返回占位符数据
    #     if self.return_paths:
    #         return img_tensor, img_tensor, img_tensor, "", "", ""
    #     else:
    #         return img_tensor, img_tensor, img_tensor, "", ""

    def __getitem__(self, index):
        pathGT = self.image_pathsGT[index]
        pathFOG = self.image_pathsFOG[index]

        imgGT = self.loader(pathGT)
        imgFOG = self.loader(pathFOG)


        ps = self.ps
        FOG_w, FOG_h = imgFOG.size

        padw = ps-FOG_w if FOG_w<ps else 0
        padh = ps-FOG_h if FOG_h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            imgFOG = TF.pad(imgFOG, (0,0,padw,padh), padding_mode='reflect')
            imgGT = TF.pad(imgGT, (0,0,padw,padh), padding_mode='reflect')


        if self.transform is not None:
            seed = np.random.randint(2147483647)  # make a seed with numpy generator
            random.seed(seed)  # apply this seed to img tranfsorms
            imgGT = self.transform(imgGT)
            random.seed(seed)  # apply this seed to img tranfsorms
            imgFOG = self.transform(imgFOG)
            random.seed(seed)


        hh, ww = imgFOG.shape[1], imgFOG.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)

        # Crop patch
        imgFOG = imgFOG[:, rr:rr+ps, cc:cc+ps]
        imgGT = imgGT[:, rr:rr+ps, cc:cc+ps]

        # print(imgFOG.shape,imgGT.shape,pathFOG)

        if self.return_paths:
            # print(pathDep,pathFOG)

            return imgGT, imgFOG, pathGT, pathFOG
        else:
            return imgGT, imgFOG,'', ''

    def __len__(self):
        return len(self.image_pathsFOG)


if __name__ == '__main__':
    path = "../Datasets/Train"
    imageFolder = ImageFolder(path)

