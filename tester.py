import os
import torch
import torchvision.utils as vutils
from readData.pairs_dataset_test import UnalignedDataLoader_test
from Models.CEDehaze import CEDehaze
import Utils
import numpy as np
import cv2
from Utils.config import Config
# from Models.Single_Bilateral import Single_Bilateralnet
# from Models.simple_est import Simple_est_Aen
# from Models.Curve_V1 import Curve_V1
# from Models.Curve_V2 import Curve_V2
# from Models.Curve_V3_ab import Curve_V3
# from Models.network_swinir import SwinIR
# from End2End.NHRN.Net import Generator
# from Models.Curve_V3 import Curve_V3
# from End2End.DehazeFormer.dehazeformer import DehazeFormer
#
# from End2End.FFA.FFA import FFA
# from End2End.Taylor.MB_TaylorFormer import MB_TaylorFormer
# from Models.ab_Curve_V3_Setting.C0_T4_f144 import   Curve_V3
from Models.Curve_V3 import Curve_V3
# from Models.ab_iterations.iter_1 import Curve_V3_iter1
# from Models.ab_iterations.iter_2 import Curve_V3
from Models.ab_iterations.iter_4 import Curve_V3_iter4
# from Models.ab_iterations.iter_6 import Curve_V3
# from Models.ab_iterations.iter_10 import Curve_V3_iter10
# from Models.ab_iterations.iter_12 import Curve_V3_iter12
# from Models.ab_iterations.iter_10 import Curve_V3_iter10
# from Models.ab_iterations.iter_12 import Curve_V3_iter12
# from Models.ab_iterations.iter_1 import Curve_V3_iter1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_gpu = torch.cuda.is_available()
opt = Config('Utils/training.yml')

data_loader = UnalignedDataLoader_test(opt)
val_loader = data_loader.load_data()
 
def HR_test():
    test_dir = os.path.join('ab_inter', 'lamb_1_nh')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    model_restoration = Curve_V3()
    model_restoration.cuda()
    # model_dir = './pretrained_model/DEHAZE/models/CEDhazeNet'
    model_dir = '/home/ps/Curve_Dehaze/Enhance_Image_Dehazing/pretrained_model1/NHHAZE/models/ab_lamb_1/'
    path_chk_rest1 = Utils.get_last_path(model_dir, '_best.pth')
    Utils.load_checkpoint(model_restoration, path_chk_rest1)
    model_restoration.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = np.zeros(len(data_loader))
    i = 0

    psnr_val_rgb = []
    for batch_idx, data_val in enumerate((val_loader), 0):
        input_ = data_val['FOG_images'].cuda()
        with torch.no_grad():
            start.record()
            restored,coeffs,aen,are = model_restoration(input_)
            # restored = model_restoration(input_)
            # restored ,coeffes = model_restoration(input_)
            end.record()
            torch.cuda.synchronize()
            time_list[i] = start.elapsed_time(end)  # milliseconds

        out_img = Utils.tensor2np(restored.detach()[0])
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        Aen = Utils.tensor2np(aen.detach()[0])
        Are = Utils.tensor2np(are.detach()[0])

        name = str(data_val['FOG_paths'][0]).split('/')[-1].split('.')[0] + '_our.png'
        imAB_gen_file = os.path.join(test_dir, '{}'.format(name))

        name_Aen = str(data_val['FOG_paths'][0]).split('/')[-1].split('.')[0] + '_Aen.png'
        imAB_Aen_file = os.path.join(test_dir, '{}'.format(name_Aen))

        name_Are = str(data_val['FOG_paths'][0]).split('/')[-1].split('.')[0] + '_Are.png'
        imAB_Are_file = os.path.join(test_dir, '{}'.format(name_Are))


        out_img = out_img / 255.0

        out_img = (out_img * 255).astype(np.uint8)

        cv2.imwrite(imAB_gen_file, out_img[:, :, [2,1,0]])

        Aen = Aen / 255.0

        Are = (Are * 255).astype(np.uint8)
        cv2.imwrite(imAB_Aen_file, Aen[:, :, [2,1,0]])
        # #
        cv2.imwrite(imAB_Are_file, Are[:, :, [2,1,0]])

        Aen_ = cv2.imread(imAB_Aen_file)
        min = Aen_.min()
        max = Aen_.max()
        # print(min,max)
        # print(Aen[0,:,:])
        # print(Are[0,:,:])
        i += 1
        print('processed item with idx: {}'.format(batch_idx))
        torch.cuda.empty_cache()

    print("-----------------------------------------------")
    print("PSNR:{}， TIME: {}ms".format(psnr_val_rgb, np.mean(time_list)))
    print("-----------------------------------------------")

if __name__ == '__main__':
    HR_test()
