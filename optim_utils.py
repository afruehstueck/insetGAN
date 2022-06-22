import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import lpips 
import numbers
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from collections import OrderedDict, namedtuple
from scipy.ndimage.filters import gaussian_filter1d

import numpy as np
from PIL import Image

class BicubicDownSample(torch.nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, device='cuda', padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32).to(device)
        k = k / torch.sum(k)
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0).to(device)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0).to(device)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            return x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x
       
def show_tensors(tensors, gh, gw, ax=None, normalize=True, title=None):
    images = tensors.clone() if torch.is_tensor(tensors) else tensors
    if len(images.shape) == 3:
        images = images.unsqueeze(1)
        
    if normalize: 
        images = 255 * ((images + 1) / 2)
    
    if torch.is_tensor(images):
        images = images.detach().clamp(0, 255).cpu().numpy()
        
    images = images.astype(np.uint8)
    tiled_images = images

    _N, C, H, W = tiled_images.shape
    tiled_images = tiled_images.reshape(gh, gw, C, H, W)
    tiled_images = tiled_images.transpose(0, 3, 1, 4, 2) #-> gh, H, gw, W, C

    tiled_images = tiled_images.reshape(gh * H, gw * W, C)
    
    if ax is None:
        fig = plt.figure(figsize=(10.24*gh, 10.24*gw), dpi=100)
        ax = plt.gca()
    else:
        ax.cla()

    plt.imshow(tiled_images)
    if title:
        plt.title(title)
    plt.axis('off')
    return tiled_images

def show_tensor(tensor, ax=None, normalize=True, text=None, color='black', size=22):
    if len(tensor.shape)==2:
        tensor = tensor.unsqueeze(0) #.repeat(3, 1, 1)
    elif len(tensor.shape)==4:
        tensor = tensor[0]
 
    if ax is None:
        fig = plt.figure(figsize=(300, 100), dpi=100)
        ax = plt.gca()
    else:
        ax.cla()
        
    tensor = tensor.detach().cpu().numpy()
    if normalize:
        tensor = 255 * ((tensor + 1) / 2)
        tensor = tensor.clip(0, 255).astype(np.uint8)
    
    image = tensor.transpose(1, 2, 0)
    ax.imshow(image)
    ax.axis('off')
    if text:
        ax.text(10, 100, text, size=size, color=color, wrap=True)#,
    return image

def tensor_to_int(tensor):
    tensor_denorm = tensor.clone()
    tensor_denorm = 255 * ((tensor_denorm + 1) / 2)
    return tensor_denorm

def save_image(tensor, filename='output', dtype='png', out_folder='', target_size=None):    
    image = tensor.clone()
    image = 255 * ((image + 1) / 2)
    image = image.squeeze().detach().clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    im = Image.fromarray(image)
    if target_size is not None:
        im = im.resize(target_size, Image.BICUBIC)
    im.save(f'{out_folder}/{filename}.{dtype}')

def save_tensor(tensor, filename='output', dtype='png', fnames=None, out_folder=''):
    num_images = tensor.shape[0]
    for t in range(num_images):
        image = tensor[t].clone()
        image = 255 * ((image + 1) / 2)
        image = image.detach().clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        im = Image.fromarray(image)
        if fnames == None:
            fname = f'{filename}_{t}.{dtype}' if num_images > 1 else f'{filename}.{dtype}'
        else: 
            fname = f'{fnames[t]}.{dtype}'
        im.save(f'{out_folder}/{fname}')
    
def get_bounding_box_face(mtcnn, human_images):
    images = tensor_to_int(human_images.clone().permute(0, 2, 3, 1))
    bounding_boxes, _ = mtcnn.detect(images)
    
    if bounding_boxes is not None and len(bounding_boxes.shape) == 3:
        bounding_boxes = bounding_boxes[0, 0, :]

    return bounding_boxes

def get_target_bounding_box_face(human_images, human_bounding_box, face_bounding_box, face_origin_size=256, vertical=True):
    if len(human_images.shape) == 3:
        human_images = human_images.unsqueeze(0)
    
    xmin, ymin, xmax, ymax = overlay_bboxes(human_bounding_box, face_bounding_box, face_origin_size=face_origin_size, vertical=vertical)
    crop_face = human_images[:, :, ymin:ymax, xmin:xmax]
    target_face = F.interpolate(crop_face, size=(256, 256), mode='bilinear', align_corners=True)#.squeeze()
    return target_face, (xmin, ymin, xmax, ymax)

# get bounding box of face region in human image based on overlaying the two bounding boxes of the faces
def overlay_bboxes(human_bounding_box, face_bounding_box, face_origin_size=256, vertical=False, align_center=False):
    xmin_human, ymin_human, xmax_human, ymax_human = human_bounding_box 
    xmid_human = int(xmax_human - (xmax_human - xmin_human) // 2)
    ymid_human = int(ymax_human - (ymax_human - ymin_human) // 2)
    xmin_face, ymin_face, xmax_face, ymax_face = face_bounding_box 
    xmid_face = int(xmax_face - (xmax_face - xmin_face) // 2)
    ymid_face = int(ymax_face - (ymax_face - ymin_face) // 2)
    
    dx_human, dy_human = xmax_human - xmin_human, ymax_human - ymin_human
    dx_face,  dy_face  = xmax_face - xmin_face,   ymax_face - ymin_face
    
    face_human_ratio = dy_human / dy_face if vertical else dx_human / dx_face
    target_size = face_origin_size * face_human_ratio 
    
    xmin_paste = max(int(xmid_human - face_human_ratio * xmid_face), 0)
    xmax_paste = min(int(xmin_paste + target_size), 1024)
    #align upper face boundary
    if align_center:   
        ymin_paste = max(int(ymid_human - face_human_ratio * ymid_face), 0)
        ymax_paste = int(ymin_paste + target_size) 
    else:
        ymin_paste = max(int(ymax_human - face_human_ratio * ymax_face), 0)
        ymax_paste = int(ymin_paste + target_size)
    
    return (xmin_paste, ymin_paste, xmax_paste, ymax_paste)

