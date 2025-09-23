import os
import torch
import random
import time
import numpy as np
import Utils
from readData.pairs_dataset import UnalignedDataLoader_train
from readData.pairs_dataset_val import UnalignedDataLoader_val
from tqdm import tqdm
import torch.optim as optim
from Models import losses
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils.visualisations import Visualizer
from Utils.config import Config
from Utils import util
from Models.SSIM import SSIM
from  Models.Single_Bilateral import Single_Bilateralnet
from Models.CEDehaze import CEDehaze
from Models.simple_est import Simple_est_Aen
from collections import OrderedDict
from Models.Curve_V1 import Curve_V1

# from Models.ab_diff_curve.low_light_curve import  Curve_V3_low_light_curve
from Models.ab_iterations.iter_1 import Curve_V3_iter1
# from Models.ab_iterations.iter_2 import Curve_V3
# from Models.ab_iterations.iter_4 import Curve_V3_iter4
# from Models.ab_iterations.iter_6 import Curve_V3
from Models.ab_iterations.iter_10 import Curve_V3_iter10
from Models.ab_iterations.iter_12 import Curve_V3_iter12
# from Models.Curve_V3_ab import Curve_V3
from Models.Curve_V3 import Curve_V3
from End2End.FFA.FFA import FFA
from End2End.Taylor.MB_TaylorFormer import MB_TaylorFormer
# from Models.network_swinir import SwinIR
# from End2End.NHRN.Net import Generator
# from End2End.Taylor.MB_TaylorFormer import MB_TaylorFormer
# from End2End.DehazeFormer.dehazeformer import DehazeFormer
# from Models.LKD import Dehaze
# from Models.swin_unet import SwinTransformerSys
opt = Config('Utils/training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# def train():
#     ######### Set Seeds ###########
#     random.seed(3407)
#     np.random.seed(3407)
#     torch.manual_seed(3407)
#     torch.cuda.manual_seed_all(3407)
#     start_epoch = 1
#
#     mode = opt.MODEL.MODE
#     session = opt.MODEL.SESSION
#
#     model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
#     Utils.mkdir(model_dir)
#
#     ######## Reveal ###########
#     visualizer = Visualizer(opt)
#
#     ####load model
#     model_restoration = MB_TaylorFormer()
#     model_restoration.cuda()
#
#     print("\n=================parameters======================")
#     print(model_restoration.parameters())
#     print(sum(param.numel() for param in model_restoration.parameters()))
#
#     new_lr = opt.OPTIM.LR_INITIAL
#     restoration_optim = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
#
#     ####### Scheduler ###########
#     # warmup_epochs = 30
#     # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer=restoration_optim,
#     #                                                         T_max=2900,
#     #                                                         eta_min=opt.OPTIM.LR_MIN,
#     #                                                         last_epoch=-1,
#     #                                                         )
#     # scheduler.step()
#
#     # scheduler = CosineAnnealingLR(restoration_optim,T_max=3000,eta_min= 1e-6)
#
#
#     ############## me
#     # scheduler = optim.lr_scheduler.StepLR(step_size=60, gamma=0.9, optimizer=restoration_optim)
#     # scheduler.step()
#
#     ###### Resume ###########
#     # if opt.TRAINING.RESUME:
#     #     path_chk_rest1 = Utils.get_last_path(model_dir, '_best.pth')
#     #     Utils.load_checkpoint(model_restoration, path_chk_rest1)
#     #     start_epoch = Utils.load_start_epoch(path_chk_rest1) + 1
#     #     Utils.load_optim(restoration_optim, path_chk_rest1)
#     #
#     #     # for i in range(1, start_epoch):
#     #     #    scheduler.step()
#     #     # new_lr = scheduler.get_last_lr()[0]
#     #     print('------------------------------------------------------------------------------')
#     #     print("==> Resuming Training with learning rate:", new_lr)
#     #     print('------------------------------------------------------------------------------')
#
#     device_ids = [i for i in range(torch.cuda.device_count())]
#     if len(device_ids) > 1:
#         print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
#         # model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
#
#     ######### Loss ###########
#     criterion_char = losses.CharbonnierLoss()
#     criterion_edge = losses.EdgeLoss()
#     criterion_ssim = SSIM()
#     criterion_tv = losses.L_TV()
#
#     ######### DataLoaders ###########
#     train_dataset = UnalignedDataLoader_train(opt)
#     train_loader = train_dataset.load_data()
#     val_dataset = UnalignedDataLoader_val(opt)
#     val_loader = val_dataset.load_data()
#
#     print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
#     print('===> Loading datasets')
#
#     best_psnr = 0
#     best_epoch = 0
#     for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
#         ######### DataLoaders ###########
#         # train_dataset = UnalignedDataLoader_train(opt)
#         # train_loader = train_dataset.load_data()
#         epoch_start_time = time.time()
#         epoch_loss = 0
#         model_restoration.train()
#
#         for i, data in enumerate(tqdm(train_loader), 0):
#
#             GT_image = data['GT_images'].cuda()
#             FOG_image = data['FOG_images'].cuda()
#             # if GT_image.shape[1] != 3:
#             #     GT_image = GT_image[:,0:3,:,:]
#             criterion_char.cuda()
#             criterion_edge.cuda()
#             criterion_ssim.cuda()
#             criterion_tv.cuda()
#
#             fine_dehaze = model_restoration(FOG_image)
#             # fine_dehaze, coeffs, Aen, Are = model_restoration(FOG_image)
#
#             #  Part1_Loss
#             # loss_ssim1 = criterion_ssim(fine_dehaze, GT_image)
#             loss_char1 = criterion_char(fine_dehaze, GT_image)
#             # loss_coffs_smooth = criterion_tv(coeffs)
#             # edgloss = criterion_edge(fine_dehaze, GT_image)
#
#             RGB_loss = loss_char1
#             # RGB_loss = (1 - loss_ssim1) + 0.1*(loss_coffs_smooth) + loss_char1
#
#
#             restoration_optim.zero_grad()
#             RGB_loss.backward()
#             restoration_optim.step()
#
#             fine_dehaze = torch.clamp(fine_dehaze, 0, 1)
#             epoch_loss += RGB_loss.item()
#             if i % 2 == 0:
#                 visualizer.display_images(images={'haze': FOG_image[0 * opt.OPTIM.BATCH_SIZE],
#                                                   # 'curve_dehaze': curve_dehaze[0 * opt.OPTIM.BATCH_SIZE],
#                                                   'fine_dehaze': fine_dehaze[0 * opt.OPTIM.BATCH_SIZE],
#                                                   # 'Aen': Aen[0 * opt.OPTIM.BATCH_SIZE],
#                                                   # 'Are': Are[0 * opt.OPTIM.BATCH_SIZE],
#                                                   'GT': GT_image[0 * opt.OPTIM.BATCH_SIZE]
#                                                   })
#         # if epoch > 6000:
#         #     scheduler.step()
#             #### Evaluation ####
#         if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
#             model_restoration.eval()
#             psnr_val_rgb = 0
#             ssim_val_rgb = 0
#             psnr_val_rgb_every = []
#             number = 0
#             for ii, data_val in enumerate((val_loader), 0):
#                 target = data_val['GT_images'].cuda()
#                 input_ = data_val['FOG_images'].cuda()
#
#                 with torch.no_grad():
#                     fine_dehaze = model_restoration(input_)
#                     # fine_dehaze, coeffs, Aen, Are = model_restoration(input_)
#                     fine_dehaze = torch.clamp(fine_dehaze, 0, 1)
#
#                 sr_img = util.tensor2np(fine_dehaze.detach()[0])
#                 gt_img = util.tensor2np(target.detach()[0])
#
#                 if ii % 1 == 0:
#                     visualizer.display_images(images={'haze': input_[0],
#                                                       # 'curve_dehaze' : curve_dehaze[0],
#                                                       'fine_dehaze': fine_dehaze[0],
#                                                       # 'mask': mask[3][0],
#                                                       'GT': target[0],
#                                                       })
#
#                 for res,tar in zip(fine_dehaze,target):
#                     psnr_val_rgb_every.append(Utils.torchPSNR(res, tar))
#                 if epoch % 10 == 0:
#                     psnr_val_rgb += util.compute_psnr(sr_img, gt_img)
#                     ssim_val_rgb += util.compute_ssim(sr_img, gt_img)
#                 number += 1
#             psnr_val_rgb_every = torch.stack(psnr_val_rgb_every).mean().item()
#             print(type(psnr_val_rgb),type(best_psnr))
#             if psnr_val_rgb_every > best_psnr:
#                 best_psnr = psnr_val_rgb_every
#                 best_epoch = epoch
#                 torch.save({'epoch': epoch,
#                             'state_dict': model_restoration.state_dict(),
#                             'MMAF_optimizer': restoration_optim.state_dict()
#                             }, os.path.join(model_dir, "best.pth"))
#
#             print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb_every, best_epoch, best_psnr))
#             if epoch % 10 == 0:
#                 avg_psnr = psnr_val_rgb / number
#                 avg_ssim = ssim_val_rgb / number
#                 torch.save({'epoch': epoch,
#                             'state_dict': model_restoration.state_dict(),
#                             'MMAF_optimizer': restoration_optim.state_dict()
#                             }, os.path.join(model_dir, "model_MMAF_epoch_"+ str(epoch) + ".pth"))
#
#                 print("[epoch %d PSNR: %.4f SSIM: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, avg_psnr, avg_ssim,
#                                                                                              best_epoch, best_psnr))
#
#         lr = restoration_optim.param_groups[0]['lr']
#         print("------------------------------------------------------------------")
#         print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, lr))
#         print("------------------------------------------------------------------")
#
#         torch.save({'epoch': epoch,
#                     'state_dict': model_restoration.state_dict(),
#                     'MMAF_optimizer': restoration_optim.state_dict()
#                     }, os.path.join(model_dir, "C_ITS.pth"))

def train():
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    start_epoch = 1

    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
    Utils.mkdir(model_dir)

    ######## Reveal ###########
    visualizer = Visualizer(opt)

    ####load model
    model_restoration = Curve_V3()
    model_restoration.cuda()

    print("\n=================parameters======================")
    print(model_restoration.parameters())
    print(sum(param.numel() for param in model_restoration.parameters()))

    new_lr = opt.OPTIM.LR_INITIAL
    restoration_optim = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ####### Scheduler ###########
    # warmup_epochs = 30
    # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(restoration_optim, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
    #                                                        eta_min=opt.OPTIM.LR_MIN)
    # scheduler = GradualWarmupScheduler(restoration_optim, multiplier=1, total_epoch=warmup_epochs,
    #                                   after_scheduler=scheduler_cosine)
    # scheduler.step()

    ############## me
    # scheduler = optim.lr_scheduler.StepLR(step_size=60, gamma=0.9, optimizer=restoration_optim)
    # scheduler.step()

    ###### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest1 = Utils.get_last_path(model_dir, '_best.pth')
        Utils.load_checkpoint(model_restoration, path_chk_rest1)
        start_epoch = Utils.load_start_epoch(path_chk_rest1) + 1
        Utils.load_optim(restoration_optim, path_chk_rest1)

        # for i in range(1, start_epoch):
        #    scheduler.step()
        # new_lr = scheduler.get_last_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
        # model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_ssim = SSIM()
    criterion_tv = losses.L_TV()

    ######### DataLoaders ###########
    train_dataset = UnalignedDataLoader_train(opt)
    train_loader = train_dataset.load_data()
    val_dataset = UnalignedDataLoader_val(opt)
    val_loader = val_dataset.load_data()

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        ######### DataLoaders ###########
        # train_dataset = UnalignedDataLoader_train(opt)
        # train_loader = train_dataset.load_data()
        epoch_start_time = time.time()
        epoch_loss = 0
        model_restoration.train()

        for i, data in enumerate(tqdm(train_loader), 0):

            GT_image = data['GT_images'].cuda()
            FOG_image = data['FOG_images'].cuda()
            # print(data['Dep_images'])
            # print(data['FOG_images'])
            # Dep_image = data['Dep_images'].cuda()
            # if GT_image.shape[1] != 3:
            #     GT_image = GT_image[:,0:3,:,:]
            criterion_char.cuda()
            criterion_edge.cuda()
            criterion_ssim.cuda()
            criterion_tv.cuda()

            # rough_dehaze,coeffs = model_restoration(FOG_image)

            # loss_ssim1 = criterion_ssim(rough_dehaze, GT_image)
            # loss_char1 = criterion_char(rough_dehaze, GT_image)
            # fine_dehaze, coeffs,rough_dehaze, Aen, Are, depth = model_restoration(FOG_image, Dep_image)

            fine_dehaze, coeffs, Aen, Are = model_restoration(FOG_image)
            loss_coffs_smooth = criterion_tv(coeffs)
            # rough_loss = 0.15*(1 - loss_ssim1) + loss_char1 + 1*(loss_coffs_smooth)
            #


            loss_ssim1 = criterion_ssim(fine_dehaze, GT_image)
            loss_char1 = criterion_char(fine_dehaze, GT_image)
            loss_edge1 = criterion_edge(fine_dehaze, GT_image)  # use for stage2
            # fine_loss = 0.    15*(1 - loss_ssim1) + loss_char1 + 1*(loss_edge1)

            RGB_loss = (1 - loss_ssim1) + loss_char1 + (loss_coffs_smooth)+loss_edge1
            # RGB_loss = (1 - loss_ssim1) + loss_char1 +  loss_edge1

            restoration_optim.zero_grad()
            RGB_loss.backward()
            restoration_optim.step()

            fine_dehaze = torch.clamp(fine_dehaze, 0, 1)
            epoch_loss += RGB_loss.item()
            if i % 10 == 0:
                visualizer.display_images(images={'haze': FOG_image[0 * opt.OPTIM.BATCH_SIZE],
                                                  # 'rough_dehaze' : rough_dehaze[0 * opt.OPTIM.BATCH_SIZE],
                                                  'fine_dehaze': fine_dehaze[0 * opt.OPTIM.BATCH_SIZE],
                                                  'Aen': Aen[0 * opt.OPTIM.BATCH_SIZE],
                                                  'Are': Are[0*opt.OPTIM.BATCH_SIZE],
                                                  'GT': GT_image[0 * opt.OPTIM.BATCH_SIZE]
                                                  })

            #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.cuda()
            model_restoration.eval()
            psnr_val_rgb = 0
            ssim_val_rgb = 0
            psnr_val_rgb_every = []
            number = 0
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val['GT_images'].cuda()
                input_ = data_val['FOG_images'].cuda()
                # depth = data_val['Dep_images'].cuda()

                with torch.no_grad():
                    # fine_dehaze, coeffs,rough_dehaze, Aen, Are,depth = model_restoration(input_,depth)
                    fine_dehaze, coeffs, Aen, Are = model_restoration(input_)
                    fine_dehaze = torch.clamp(fine_dehaze, 0, 1)

                sr_img = util.tensor2np(fine_dehaze.detach()[0])
                gt_img = util.tensor2np(target.detach()[0])

                if ii % 10 == 0:
                    visualizer.display_images(images={'haze': input_[0],
                                                      # 'rough_dehaze' : rough_dehaze[2][0],
                                                      'fine_dehaze': fine_dehaze[0],
                                                      # 'mask': mask[2][0],
                                                      'GT': target[0],
                                                      })

                for res,tar in zip(fine_dehaze,target):
                    psnr_val_rgb_every.append(Utils.torchPSNR(res, tar))

                if epoch % 10 == 0:
                    psnr_val_rgb += util.compute_psnr(sr_img, gt_img)
                    ssim_val_rgb += util.compute_ssim(sr_img, gt_img)
                number += 1

            psnr_val_rgb_every = torch.stack(psnr_val_rgb_every).mean().item()

            if psnr_val_rgb_every > best_psnr:
                best_psnr = psnr_val_rgb_every
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'MMAF_optimizer': restoration_optim.state_dict()
                            }, os.path.join(model_dir, "model_MMAF_best.pth"))

            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb_every, best_epoch, best_psnr))
            if epoch % 10 == 0:
                avg_psnr = psnr_val_rgb / number
                avg_ssim = ssim_val_rgb / number
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'MMAF_optimizer': restoration_optim.state_dict()
                            }, os.path.join(model_dir, "model_MMAF_epoch_"+ str(epoch) + ".pth"))

                print("[epoch %d PSNR: %.4f SSIM: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, avg_psnr, avg_ssim,
                                                                                             best_epoch, best_psnr))
        # scheduler.step()
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, new_lr))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'MMAF_optimizer': restoration_optim.state_dict()
                    }, os.path.join(model_dir, "C_ITS.pth"))

if __name__ == '__main__':
    train()
