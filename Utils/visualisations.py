import numpy as np
import torch
from visdom import Visdom

tensor2image = lambda T: (127.5 * (T[0].cpu().float().numpy() + 1.0)).astype(np.uint8)


def tensor2im(input_imagae, imtype=np.uint8):

    if not isinstance(input_imagae, imtype):
        if isinstance(input_imagae, torch.Tensor):
            image_tensor = input_imagae.data
        else:
            return input_imagae
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 3, 0)) + 1) / 3.0 * 255.0
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:
        image_numpy = input_imagae
    return image_numpy.astype(imtype)


class Visualizer(object):
    def __init__(self, opt):
        self.viz = Visdom()
        self.image_windows = {}
        self.display_id = opt.TRAINING.DISPLAY_ID
        self.title = 'training loss {}'.format(opt.TRAINING.BACKWARD_TYPE)
        if opt.TRAINING.DISPLAY_ID > 0:
            import visdom
            self.vis1 = visdom.Visdom()
            self.vis2 = visdom.Visdom()

    def display_images(self, images=None):
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2im(tensor).transpose([2, 0, 1]),
                                                                opts={'title': image_name})
            else:
                self.viz.image(tensor2im(tensor).transpose([2, 0, 1]), win=self.image_windows[image_name],
                               opts={'title': image_name})

    def plot_errors(self, errors, epoch, fraction_passed):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + fraction_passed)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis1.line(
            X=np.stack([self.plot_data['X']] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.title,
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id
        )
        # return self.plot_data['X'][-1]

    def plot_PSNR(self, psnr, epoch, fraction_passed):
        if not hasattr(self, 'plot_psnr'):
            self.plot_psnr = {'X': [], 'Y': [], 'legend': list(psnr.keys())}
        self.plot_psnr['X'].append(epoch + fraction_passed)
        self.plot_psnr['Y'].append([psnr[k] for k in self.plot_psnr['legend']])
        self.vis2.line(
            X=np.stack([self.plot_psnr['X']] * len(self.plot_psnr['legend']), 1),
            Y=np.array(self.plot_psnr['Y']),
            opts={
                'title': 'val psnr value' ,
                'legend': self.plot_psnr['legend'],
                'xlabel': 'epoch',
                'ylabel': 'psnr'},
            win= self.display_id
        )
