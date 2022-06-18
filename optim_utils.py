import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import lpips 

import numbers
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import collections
from collections import OrderedDict, namedtuple
from scipy.ndimage.filters import gaussian_filter1d

import numpy as np
from PIL import Image
        
class Images(Dataset):
    def __init__(self, root_dir):
        self.root_path = Path(root_dir)
        self.image_list = sorted(list(self.root_path.glob("*.png")))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        return image,img_path.stem

class BicubicDownSample(nn.Module):
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
            
def plot_losses(input_loss_dict, ax=None, title='', scale_losses=None, smoothing=True, smoothing_sigma=7, cmap='CMRmap'): #cmap = 'turbo', 'Spectral'
    if ax is None:
        fig = plt.figure(figsize=(10.24*16, 10.24*10), dpi=100)
        ax = plt.gca()
        
    loss_dict = {}
    labels = {}
    max_num_epochs = 0
    
    for loss_name, losses in input_loss_dict.items():
        if type(losses) == collections.defaultdict: # recursive plotting for subdicts
            plot_losses(losses, ax=ax, title=loss_name, scale_losses=scale_losses, smoothing=smoothing, smoothing_sigma=smoothing_sigma, cmap=cmap)
        else:
            max_num_epochs = max(max_num_epochs, len(losses))
            
            label = f'{loss_name}'
            if scale_losses and loss_name in scale_losses:
                lambda_loss = scale_losses[loss_name]
                losses = np.array(losses) * lambda_loss
                label += f' | λ = {lambda_loss}'
            
            labels[loss_name] = label
            loss_dict[loss_name] = losses
        
    num = len(loss_dict.keys())
    
    # get num colors from color map
    color_map = cm.get_cmap(cmap) 
    colors = [ color_map(i) for i in np.arange(0, 1, 1/num) ]
        
    for idx, (loss_name, losses) in enumerate(loss_dict.items()):
        x = np.linspace(0, max_num_epochs, len(losses), endpoint=False) #scales x-axis for elements that don't happen every epoch (regularizer)
        ax.plot(x, losses, color=colors[idx], alpha=0.25 if smoothing else 1.0, label=labels[loss_name] if not smoothing else '')
        
    if smoothing:    
        for idx, (loss_name, losses) in enumerate(loss_dict.items()):
            x = np.linspace(0, max_num_epochs, len(losses), endpoint=False) #scales x-axis for elements that don't happen every epoch (regularizer)
            smooth_losses = gaussian_filter1d(np.array(losses), sigma=smoothing_sigma)
            ax.plot(x, smooth_losses, color=colors[idx], label=labels[loss_name])
            
            if loss_name == 'total' and len(smooth_losses)>5:
                ax.set_ylim([0, np.max(smooth_losses[5:])])
    
    ax.axis('tight')
    #ax.relim()
    #ax.autoscale_view()
    #ax.title(title)
    ax.legend()
    #plt.show()
    
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

def plot_rectangle(axes, box, color=(0.92, 0.35, 0.04)):
    box = box[0] if len(box) == 1 else box
    rect = patches.Rectangle((int(box[0]), int(box[1])), int(box[2])-int(box[0]), int(box[3])-int(box[1]), linewidth=3, edgecolor=color, facecolor='none')
    axes.add_patch(rect)

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

def update_figure(tensor, ax, normalize=True, text=None, color='white', size=14):
    ax.cla() 
    show_tensor(tensor, ax, normalize, text, color, size)

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
            
def rgb2gray(rgb):
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def get_segmentation_bbox(segmentation, value):
    c, h, w = segmentation.shape
    selection = (segmentation == value).all(axis=0)
    rows, cols = torch.any(selection, dim=1), torch.any(selection, dim=0)
    
    ymin, ymax = (0, h) if not torch.any(rows) else torch.where(rows)[0][[0, -1]].detach().cpu().numpy()
    xmin, xmax = (0, w) if not torch.any(cols) else torch.where(cols)[0][[0, -1]].detach().cpu().numpy()
    return np.array([xmin, ymin, xmax, ymax]).astype(np.float)

# get segmentation mask based on face input image
def get_segmentations_faces(input_images, bisenet):
    target_shape = input_images.shape[-2:]
    
    if input_images.shape[-1] != 512:
        input_images = F.interpolate(input_images, size=(512, 512), mode='bilinear', align_corners=True) 
            
    segmentations = bisenet(input_images)[0].argmax(1).unsqueeze(1).float()
    segmentations = F.interpolate(segmentations, size=target_shape, mode='nearest').int()
       
    bboxes = []
    for segmentation in segmentations:
        face_bbox = get_segmentation_bbox(segmentation, value=1) #face segment has value 1
        hair_bbox = get_segmentation_bbox(segmentation, value=17) #hair segment has value 1
        face_bbox[1] = min(hair_bbox[1], face_bbox[1]) #use upper boundary of hair for y bounds
        bboxes.append(face_bbox)
        
    if len(input_images) == 1: #reduce dimensions if input is just 1 image
        segmentations = segmentations[0]
        bboxes = bboxes[0]
    return segmentations, bboxes

# get segmentation mask based on input image
def get_segmentations_humans(input_images, #full body human images to be segmented
                             bisenet, #segmentation network
                             human_seg_areas=(402, 0, 622, 220), #area in human image to be used for face segmentation
                             face_target_boxes = None, #segmentation boxes from face images to be aligned to
                             target_w = 256
                            ):
    if len(input_images.shape) == 3:
        input_images = input_images.unsqueeze(0)
    
    human_seg_contexts = []
    if type(human_seg_areas) is tuple:
        # use an approximate face region to get segmentations from
        human_seg_contexts = F.interpolate(input_images[:, :, human_seg_areas[1]:human_seg_areas[3], human_seg_areas[0]:human_seg_areas[2]], size=(512, 512), mode='bilinear', align_corners=True)
        # expand for batch processing
        human_seg_areas = ((human_seg_areas, ) * len(input_images))
    else:
        # use exact bounding boxes for more precise face regions
        for input_image, (xmin_seg_area, ymin_seg_area, xmax_seg_area, ymax_seg_area) in zip(input_images, np.array(human_seg_areas).astype(np.int32)):
            human_seg_contexts.append(F.interpolate(input_image[:, ymin_seg_area:ymax_seg_area, xmin_seg_area:xmax_seg_area].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True).squeeze())
        human_seg_contexts = torch.stack(human_seg_contexts)
    
    # get face segmentations from cropped context images
    human_segmentations, human_context_face_bboxes = get_segmentations_faces(human_seg_contexts, bisenet)
    
    if len(human_seg_contexts) == 1: # unsqueeze for batch processing
        human_context_face_bboxes = ((human_context_face_bboxes, ) * len(human_seg_contexts))
        face_target_boxes = ((face_target_boxes, ) * len(human_seg_contexts))
        
    human_face_bboxes = [] #collect face bounding boxes in human space 
    face_crop_bboxes = []
    human_target_bboxes = []
    if face_target_boxes is not None:
        for human_seg_area, context_face_bbox, target_cropbox in zip(human_seg_areas, human_context_face_bboxes, face_target_boxes):
            #initial region crop for segmentation in human image
            xmin_seg_area, ymin_seg_area, xmax_seg_area, ymax_seg_area = human_seg_area 
            segmentation_ratio = (ymax_seg_area - ymin_seg_area) / 512
            
            #box of face region within segmentation mask (in cropped context)
            xmin_face, ymin_face, xmax_face, ymax_face = context_face_bbox
            h_face, w_face = ymax_face - ymin_face, xmax_face - xmin_face
            xmid_face = xmin_face + w_face / 2
            
            #crop box of target segmentation
            xmin_target, ymin_target, xmax_target, ymax_target = target_cropbox 
            h_target, w_target = ymax_target - ymin_target, xmax_target - xmin_target
            ratio = h_face / h_target
            
            # create new crop box for human face based on target box
            ymin_crop_face = ymin_face - ratio * ymin_target
            ymax_crop_face = ymax_face + ratio * (target_w-ymax_target)
            xmin_crop_face = xmid_face - ratio * (target_w//2) #xmin_face - ratio * xmin_target
            xmax_crop_face = xmin_crop_face + (ymax_crop_face-ymin_crop_face)
               
            human_face_bboxes.append(np.array([xmin_seg_area, ymin_seg_area, xmin_seg_area, ymin_seg_area]) + np.array([xmin_face, ymin_face, xmax_face, ymax_face])*segmentation_ratio)
            
            face_crop_box = np.array((xmin_crop_face, ymin_crop_face, xmax_crop_face, ymax_crop_face))
            face_crop_bboxes.append(face_crop_box)
            
            crop_box = np.array([xmin_seg_area, ymin_seg_area, xmin_seg_area, ymin_seg_area]) + face_crop_box*segmentation_ratio
            h_crop = crop_box[3] - crop_box[1] 
            w_crop = crop_box[2] - crop_box[0]
            assert w_crop != 0
            crop_box[1] = max(crop_box[1], 0) #ensure crop box doesn't get negative
            crop_box[3] = crop_box[1] + w_crop
            human_target_bboxes.append(crop_box)
            
    if len(input_images) == 1: #reduce dimensions if input is just 1 image
        human_segmentations = human_segmentations[0]
        human_context_face_bboxes = human_context_face_bboxes[0]
        human_face_bboxes = human_face_bboxes[0]
        face_crop_bboxes = face_crop_bboxes[0]
        human_target_bboxes = human_target_bboxes[0]
        
    return human_segmentations, human_context_face_bboxes, human_face_bboxes, face_crop_bboxes, human_target_bboxes

def expand_face_regions(bboxes, factor=1.5):
    regions = []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        w, h = xmax - xmin, ymax - ymin
        
        xmid = int(xmin + w // 2)
        ymid = int(ymin + h // 2)

        face_sz = min(min(int(factor * max(h, w)), ymid), xmid)
        xmin_crop, ymin_crop, xmax_crop, ymax_crop = xmid-face_sz, ymid-face_sz, xmid+face_sz, ymid+face_sz
        
        regions.append((xmin_crop, ymin_crop, xmax_crop, ymax_crop))
    return regions

# get face region based on MTCNN face bounding box detector
def get_face_regions_mtcnn(input_images, mtcnn):
    mtcnn_bboxes = get_bounding_box_MTCNN(mtcnn, input_images) 
    face_boxes = [bbox[0] for bbox in mtcnn_boxes]
    return expand_face_regions(face_boxes)


def get_landmarks_MobileNet(mobilenet, human_images):
    images = tensor_to_int(human_images.clone().permute(0, 2, 3, 1))
    bounding_boxes, _ = mtcnn.detect(images)

    if bounding_boxes is not None and len(bounding_boxes.shape) == 3:
        bounding_boxes = bounding_boxes[0, 0, :]
    else:
        bounding_boxes = None
        
    return bounding_boxes

def get_boxes_and_landmarks_MTCNN_batch(mtcnn, images):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    images = tensor_to_int(images.clone().permute(0, 2, 3, 1))

    bounding_boxes, probabilities,landmarks = mtcnn.detect(images, landmarks=True)
    
    if bounding_boxes is not None:
            output_boxes = []
            output_landmarks = []
            output_landmark_boxes = []
            for bounding_box, landmark in zip(bounding_boxes, landmarks):
                if bounding_box is not None:
                    bounding_box = bounding_box[0]
                    landmark = landmark[0]
                    #bounding_box = bounding_box.squeeze()
                    #landmark = landmark.squeeze()
                    #print(bounding_box.shape)
                    #print(landmark.shape)
                    if len(bounding_box.shape) > 1:
                        bounding_box = bounding_box[:, 0, :] #assuming only one face per image detected
                        landmark = landmark[:, 0, :] #assuming only one face per image detected
                    output_boxes.append(bounding_box)
                    output_landmarks.append(landmark)

                    #lm_bboxes = []
                    #for lm in landmark: #calculate bounding boxes of landmarks - todo vectorize
                    #    lm_bboxes.append(np.array([np.min(lm[:, 0]), np.min(lm[:, 1]), np.max(lm[:, 0]), np.max(lm[:, 1])]))
                    #if len(lm_bboxes) == 1: #reduce dimension for single input
                    #    lm_bboxes = lm_bboxes[0]
                    output_landmark_boxes.append(np.array([np.min(landmark[:, 0]), np.min(landmark[:, 1]), np.max(landmark[:, 0]), np.max(landmark[:, 1])]))
                else:
                    output_boxes.append(None)#[]
                    output_landmarks.append(None) #= []
                    output_landmark_boxes.append(None) #= []
                    #print(bounding_box)
                    #assert 0
                    #print('error!')
                    #return None, None, None
            if len(output_boxes) == 1:
                output_boxes = output_boxes[0]
                output_landmarks = output_landmarks[0]
                output_landmark_boxes = output_landmark_boxes[0]
            return output_boxes, output_landmarks, output_landmark_boxes
        
    else:
        return None, None, None #bounding_boxes, probabilities, landmarks #

    
def get_boxes_and_landmarks_MTCNN(mtcnn, images):
    images = tensor_to_int(images.clone().permute(0, 2, 3, 1))
    #print(images.shape)
    bounding_boxes, probabilities, landmarks = mtcnn.detect(images, landmarks=True)
    #print(bounding_boxes.shape)
    if bounding_boxes is not None and len(bounding_boxes.shape) == 3:
        #print('foo')
        bounding_boxes = bounding_boxes[:, 0, :] #assuming only one face per image detected
        landmarks = landmarks[:, 0, :] #assuming only one face per image detected
        lm_bboxes = []
        for lm in landmarks: #calculate bounding boxes of landmarks - todo vectorize
            lm_bboxes.append(np.array([np.min(lm[:, 0]), np.min(lm[:, 1]), np.max(lm[:, 0]), np.max(lm[:, 1])]))
        if len(lm_bboxes) == 1: #reduce dimension for single input
            lm_bboxes = lm_bboxes[0]
#         else:
#             print('bb ', bounding_boxes.shape)
#             print('lm ', landmarks.shape)
#             lm_boxes = np.array([np.min(landmarks[:, 0]), np.min(landmarks[:, 1]), np.max(landmarks[:, 0]), np.max(landmarks[:, 1])])
        return bounding_boxes.squeeze(), landmarks.squeeze(), lm_bboxes
    else:
        return bounding_boxes, probabilities, landmarks #None, None, None

def get_bounding_box_MTCNN(mtcnn, human_images):
    images = tensor_to_int(human_images.clone().permute(0, 2, 3, 1))
    bounding_boxes, _ = mtcnn.detect(images)
    
    if bounding_boxes is not None and len(bounding_boxes.shape) == 3:
        bounding_boxes = bounding_boxes[0, 0, :]
    #else:
    #    bounding_boxes = None
        
    return bounding_boxes

# get bounding box of face region in human image based on overlaying the two bounding boxes of the faces
def overlay_bboxes(human_bounding_box, face_bounding_box, face_origin_size=256, vertical=False, align_center=False):
    xmin_human, ymin_human, xmax_human, ymax_human = human_bounding_box 
    xmid_human = int(xmax_human - (xmax_human - xmin_human) // 2)
    ymid_human = int(ymax_human - (ymax_human - ymin_human) // 2)
#    ymin_human = ymin_human - 10
    xmin_face, ymin_face, xmax_face, ymax_face = face_bounding_box 
    xmid_face = int(xmax_face - (xmax_face - xmin_face) // 2)
    ymid_face = int(ymax_face - (ymax_face - ymin_face) // 2)
    
    #print(f'human: {human_bounding_box}')
    
    #print(f'face: {face_bounding_box}')
    
    dx_human, dy_human = xmax_human - xmin_human, ymax_human - ymin_human
    dx_face,  dy_face  = xmax_face - xmin_face,   ymax_face - ymin_face
    
    face_human_ratio = dy_human / dy_face if vertical else dx_human / dx_face
    target_size = face_origin_size * face_human_ratio 
    
    xmin_paste = max(int(xmid_human - face_human_ratio * xmid_face), 0)
    xmax_paste = min(int(xmin_paste + target_size), 1024)
    
#     # align upper face boundary
    if align_center:   
        ymin_paste = max(int(ymid_human - face_human_ratio * ymid_face), 0)
        ymax_paste = int(ymin_paste + target_size) 
    else:
        ymin_paste = max(int(ymax_human - face_human_ratio * ymax_face), 0)
        ymax_paste = int(ymin_paste + target_size)
    
    #print(f'paste: {xmin_paste} {ymin_paste} {xmax_paste} {ymax_paste}')
    return (xmin_paste, ymin_paste, xmax_paste, ymax_paste)


def get_target_MTCNN(human_images, human_bounding_box, face_bounding_box, face_origin_size=256, vertical=True):
    if len(human_images.shape) == 3:
        human_images = human_images.unsqueeze(0)
    
    #xmin, ymin, xmax, ymax = overlay_bboxes_old(human_bounding_box, face_bounding_box, face_origin_size=face_origin_size, vertical=vertical)
    xmin, ymin, xmax, ymax = overlay_bboxes(human_bounding_box, face_bounding_box, face_origin_size=face_origin_size, vertical=vertical)
    crop_face = human_images[:, :, ymin:ymax, xmin:xmax]
    target_face = F.interpolate(crop_face, size=(256, 256), mode='bilinear', align_corners=True)#.squeeze()
    return target_face, (xmin, ymin, xmax, ymax)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        h = self.kernel_size // 2
        input = F.pad(input, (h, h, h, h), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)
    
