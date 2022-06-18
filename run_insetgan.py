import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.nn import functional as F
from torchvision import utils, transforms
from pathlib import Path
import PIL
from math import log10, ceil
import numpy as np
import os
import lpips
import pickle
from collections import defaultdict
from optim_utils import *

#from face_extractor import *

from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import time

#import matplotlib.patches as patches
import dnnlib
import contextlib
from facenet_pytorch import MTCNN
from torch.autograd import Variable as V

home_dir = '.'
device = torch.device('cuda:0')

out_folder = f'{home_dir}/results'
os.makedirs(out_folder, exist_ok=True)

#############################################################################################
# LOAD HUMAN NETWORK
#############################################################################################
ckpt = f'{home_dir}/networks/DeepFashion_1024x768.pkl'

with open(ckpt, 'rb') as f:
    networks = pickle.Unpickler(f).load()   

G_human = networks['G_ema'].to(device)

# define average human
w_samples = G_human.mapping(torch.from_numpy(np.random.RandomState(123).randn(10000, G_human.z_dim)).to(device), None) 
w_samples = w_samples[:, :1, :]
latent_avg_human = torch.mean(w_samples, axis=0).squeeze()
latent_avg_human = latent_avg_human.unsqueeze(0).repeat(G_human.num_ws, 1).unsqueeze(0)   

print('... loaded canvas generator.')

#############################################################################################
# LOAD FACE NETWORK
#############################################################################################
ckpt = f'{home_dir}/networks/ffhq.pkl'

with open(ckpt, 'rb') as f:
    networks = pickle.Unpickler(f).load()   

G_face = networks['G_ema'].to(device)

# define average face
w_samples = G_face.mapping(torch.from_numpy(np.random.RandomState(123).randn(10000, G_face.z_dim)).to(device), None) 
w_samples = w_samples[:, :1, :]
latent_avg_face = torch.mean(w_samples, axis=0).squeeze()
latent_avg_face = latent_avg_face.unsqueeze(0).repeat(G_face.num_ws, 1).unsqueeze(0)   

print('... loaded inset generator.')

#############################################################################################
# LOAD FACE LOCATOR NETWORK (FACENET_PYTORCH)
#############################################################################################
mtcnn = MTCNN(device=device, keep_all=False, select_largest=False)
mtcnn.requires_grad = False

print('... loaded face locator.')

#############################################################################################
# LOSS FUNCTIONS
#############################################################################################
percept = lpips.LPIPS(net='vgg').to(device)

loss_L1 = torch.nn.L1Loss().to(device) 
loss_L2 = torch.nn.MSELoss().to(device)
loss_FN = loss_L1

#image_res = 256

def rgb2gray(rgb):
    r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def l1_loss(target, optim):
    res = float(target.shape[-1])
    return loss_L1(target, optim) / ( res ** 2 )

def l2_loss(target, optim):
    res = float(target.shape[-1])
    return loss_L2(target, optim) / ( res ** 2 ) 

def percep_loss(target, optim):
    return percept(target, optim).sum()

def bottom_loss(target, optim, edge=16):
    return percept(target[:, :, -edge:, 16:-16], optim[:, :, -edge:, 16:-16]).sum()

def edge_loss(target, optim, edge=8, bottom_multiplier=1): # weight lower edge more
    res = float(target.shape[-1])
    return  loss_L2(target[:, :, :edge, :], optim[:, :, :edge, :]) / (edge*res) + \
            loss_L2(target[:, :, -edge:, :], optim[:, :, -edge:, :]) / (edge*res) * bottom_multiplier + \
            loss_L2(target[:, :, edge:-edge, :edge], optim[:, :, edge:-edge, :edge]) / (edge*(res-2*edge)) + \
            loss_L2(target[:, :, edge:-edge, -edge:], optim[:, :, edge:-edge, -edge:]) / (edge*(res-2*edge))  #normalize per pixel

def percep_edge_loss(target, optim, edge=8, bottom_multiplier=1): # weight lower edge more
    target_cp = target.clone()
    optim_cp = optim.clone()
    target_cp[:, :, edge:-edge, edge:-edge] = 0
    optim_cp[:, :, edge:-edge, edge:-edge] = 0
    return  percep_loss(target_cp, optim_cp)

def disc_loss(optim):
    with torch.no_grad():
            disc = D_human(optim, None)
    return torch.nn.functional.softplus(-disc).squeeze() 

def pose_loss(target_pose, pose):
    loss = torch.sum(pose.squeeze()[:19] - target_pose.squeeze()[:19] ** 2)
    return loss

def regularize_loss(latent):
    if in_W_space:
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - X_mean).bmm(X_comp.T.unsqueeze(0)) / X_stdev
    else:
        latent_p_norm = latent
    return latent_p_norm.pow(2).mean()

def ssim_loss(target, optim):
    target_gray = rgb2gray((target + 1) / 2.)
    optim_gray = rgb2gray((optim + 1) / 2.)

    loss = 1.0 - ssim(target_gray, optim_gray, data_range=1, size_average=True)

    return loss

def bounding_box_distance(box_old, box_new):
    return l2_loss(box_old, box_new)    

def mean_latent_loss(w, w_avg):
    return l2_loss(w, w_avg)    

seed_canvas = 1111 
z = torch.from_numpy(np.random.RandomState(seed_canvas).randn(16, G_human.z_dim)).to(device)
trunc_canvas = 0.4

with torch.no_grad():
    random_humans_w = G_human.mapping(z, None) 
    random_humans_w = random_humans_w * (1 - trunc_canvas) + latent_avg_human * trunc_canvas
with torch.no_grad():
    random_outputs = G_human.synthesis(random_humans_w.to(device), noise_mode='const') 

save_tensor(random_outputs, 'human', out_folder=out_folder)


seed_inset = 1111 
z = torch.from_numpy(np.random.RandomState(seed_inset).randn(16, G_face.z_dim)).to(device)
trunc_inset = 0.4
trunc_insets = [0.99, 0.9, 0.85, 0.8, 0.7, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
with torch.no_grad():
    random_face_w = G_face.mapping(z, None) 
    print(f'random_face_w: {random_face_w.shape}')
    print(f'latent_avg_face: {latent_avg_face.shape}')
    for i in range(18):
        random_face_w[:, i, :] = random_face_w[:, i, :] * (1 - trunc_insets[i]) + latent_avg_face[:, i, :] * trunc_insets[i]
with torch.no_grad():
    random_outputs = G_face.synthesis(random_face_w.to(device), noise_mode='const') 

save_tensor(random_outputs, 'face', out_folder=out_folder)


data_transforms = transforms.Compose([
    transforms.ToTensor()
])


plt.figure(figsize=(14, 14))

face_res = G_face.synthesis.img_resolution 

down_res = 64
downsampler_256 = BicubicDownSample(factor=1024//256, device=device)
downsampler_128 = BicubicDownSample(factor=1024//128, device=device)
downsampler_64 = BicubicDownSample(factor=1024//down_res, device=device)
downsampler = BicubicDownSample(factor=256//down_res, device=device)

loss_fn_dict = {
    'L2':                 l2_loss,
    'L1':                 l1_loss,
    'L1_gradient':        l1_loss,
    'L1_in':               l1_loss,
    'perceptual':         percep_loss,
    'perceptual_in':      percep_loss,
    'perceptual_face':      percep_loss,
    'perceptual_edge':    percep_edge_loss,
    'edge':               edge_loss,
    'mean_latent':        mean_latent_loss,
    'selected_body':      percep_loss,
    'selected_body_L1':   l1_loss,
}

loss_fn_downsample = ['L1', 'perceptual'] 

lambdas_w_face = {
    'L1': 500, #
    'perceptual_in': 0.55,
    'perceptual': 0.05,  
    'perceptual_edge': 0.1, 
    'edge': 40000, 
}

lambdas_w_human = {
    'L1': 500, 
    'perceptual': 0.15, 
    'edge': 2500,
    'mean_latent': 25000,
    'selected_body': 0.01,
    'selected_body_L1': 1000,
}


clothing  = random_humans_w[[0, 1]]
faces = random_face_w[[0, 1, 3, 4, 5, 8, 12]]

for person in range(len(faces)): 
    for body in [0]:
        
        latent_w_human = clothing[body][0].detach().clone()
        latent_w_face = faces[person].unsqueeze(0).detach().clone()
        losses_w_face = defaultdict(list)
        losses_w_human = defaultdict(list)
        latent_in_human = latent_w_human.unsqueeze(0).unsqueeze(0).repeat(1, G_human.num_ws, 1).to(device)
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad():
                gen_human = G_human.synthesis(latent_in_human, noise_mode='const')   
        
        optim_w = []     

        ctx_frame = 16

                ################################################################################################################
        # find targets
        ################################################################################################################

        #latent_w_face = latent_w_face.unsqueeze(0).unsqueeze(0).repeat(1, G_face.num_ws, 1).to(device)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad():
                hires_face = G_face.synthesis(latent_w_face, noise_mode='const') 
            save_image(hires_face, f'{person}_face_input', out_folder=out_folder)
        
        if face_res == 1024:
            input_face = downsampler_256(hires_face) 
        else:
            input_face = hires_face
            
        face_bounding_box = get_bounding_box_MTCNN(mtcnn, input_face)
        input_downsampled = ( downsampler_64(hires_face) if face_res == 1024 else downsampler(hires_face) )

        human_bounding_box = get_bounding_box_MTCNN(mtcnn, gen_human)
        
        gen_face, crop_box = get_target_MTCNN(gen_human, human_bounding_box.squeeze(), face_bounding_box.squeeze(), vertical=False)
        xmin, ymin, xmax, ymax = crop_box

        w_face = xmax - xmin 
        h_face = ymax - ymin

        latent_w_face_input = latent_w_face.clone()
        latent_in = latent_w_human.unsqueeze(0).repeat(G_human.num_ws, 1).unsqueeze(0)   

        opt_human = torch.optim.Adam([latent_w_human], lr=0.035)
        opt_face = torch.optim.Adam([latent_w_face], lr=0.005)

        #target_downsampled = input_downsampled.clone()

        optim_human, optim_face = True, False #, 
        best_face_state, best_human_state = None, None
        best_human_loss, best_face_loss = 1000000, 1000000

        selected_body = None
        pbar = tqdm(range(400), position=1, leave=True)
        for j in pbar: #range(750):
            latent_w_face.requires_grad = optim_face
            latent_w_human.requires_grad = optim_human

            optimizer = opt_face if optim_face else opt_human
            optimizer.zero_grad()                
            
            if j == 75:
                selected_body = downsampler_128(gen_human).squeeze() 
        
            
            latent_in = latent_w_human.unsqueeze(0).repeat(G_human.num_ws, 1).unsqueeze(0) 
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                gen_human = G_human.synthesis(latent_in, noise_mode='const') 


            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                hires_face = G_face.synthesis(latent_w_face, noise_mode='const')

            if face_res == 1024:
                target_face = downsampler_256(hires_face) #256 x 256 face
            else:
                target_face = hires_face
            target_downsampled = downsampler(target_face) #64 x 64 face 
                 
            if j % 25 == 0 and j <= 100:
                new_human_bounding_box = get_bounding_box_MTCNN(mtcnn, gen_human)
                xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox = new_human_bounding_box
                if ( new_human_bounding_box is not None 
                and ymax_bbox - ymin_bbox >= 90 and ymax_bbox - ymin_bbox <= 125  # prevent face from becoming too small or large
                and ymin_bbox > 6 and ymin_bbox < 64  # bbox should stay close to upper edge of image
                and abs(xmin_bbox - 512) < 200 ):  # bbox should stay close to center of image
                    delta_bounding_box = new_human_bounding_box-human_bounding_box
                    human_bounding_box = human_bounding_box + ((250-j)/250) * delta_bounding_box #new_human_bounding_box

            gen_face, crop_box = get_target_MTCNN(gen_human, human_bounding_box.squeeze(), face_bounding_box.squeeze(), vertical=False)
            xmin, ymin, xmax, ymax = crop_box

            gen_downsampled = downsampler(gen_face) #64 x 64 human face region

            # accumulate losses
            total_loss = 0        
            loss_dict = losses_w_face if optim_face else losses_w_human
            loss_source = lambdas_w_face if optim_face else lambdas_w_human

            loss_info_names = []
            loss_info_quantities = []
            for loss_name in loss_source.keys():  
                loss = 0
                loss_info_names.append(loss_name)
                loss_weight = loss_source[loss_name]

                if 'edge' in loss_name and j <= 300: # slowly increase edge loss influence
                    loss_weight = loss_weight * max(j - 100, 0) / 200

                loss_fn = loss_fn_dict[loss_name]

                if face_bounding_box is not None:
                    xmin_face, ymin_face, xmax_face, ymax_face = face_bounding_box
                    xmin_face_down, ymin_face_down, xmax_face_down, ymax_face_down = np.array([xmin_face, ymin_face, xmax_face, ymax_face], dtype=np.uint8) // 4
                    xdiff = xmax_face - xmin_face
                    xmid_face = int(xmin_face + float(xdiff / 2))
                    xdiff_mult4 = ((16 * round(xdiff/16))) // 2 + 16
                    xmin_face_border = max(xmid_face - xdiff_mult4, 0)
                    xmax_face_border = min(xmid_face + xdiff_mult4, 256)
                    ymax_face = int(ymax_face)
                if optim_face:
                    if loss_name == 'perceptual_in': #constrain interior region of image stay close to input face
                        if input_downsampled is not None:
                            loss = loss_fn(input_downsampled[:, :, 2:ymax_face_down-2, xmin_face_down+2:xmax_face_down-2], target_downsampled[:, :, 2:ymax_face_down-2, xmin_face_down+2:xmax_face_down-2])
                        else:
                            loss = torch.zeros_like(total_loss).to(device)
                    else:
                        if loss_name in loss_fn_downsample:
                            t, g = target_downsampled, gen_downsampled 
                        else:
                            t, g = target_face, gen_face
                        loss = loss_fn(t, g)
                else: # optim_human
                    if loss_name == 'mean_latent':
                        loss = loss_fn(latent_in, latent_avg_human.clone())    
                    elif 'selected_body' in loss_name:
                        if selected_body is not None:
                            downsampled_body = downsampler_128(gen_human)
                            t = V(selected_body[:, 28:, 32:-32])
                            g = downsampled_body[:, :, 28:, 32:-32]
                            loss = loss_fn(t, g)   
                        else:
                            loss = torch.zeros_like(total_loss).to(device)
                    else:
                        t = target_downsampled if loss_name in loss_fn_downsample else target_face
                        g = gen_downsampled if loss_name in loss_fn_downsample else gen_face
                        #if j < 250:
                        #    t = rgb2gray(t)
                        #    g = rgb2gray(g)
                        loss = loss_fn(t, g)
                total_loss += loss_weight * loss

                loss_val = loss.clone().detach().cpu().numpy()
                loss_info_quantities.append(loss_weight * loss_val)
                loss_dict[loss_name].append(loss_val)

            total_loss.backward()
            optimizer.step()

            if optim_face:
                if (j > 0 and losses_w_face['edge'][-1] < best_face_loss and losses_w_human['edge'][-1] <= best_human_loss if 'edge ' in losses_w_human else True):
                    best_face_loss = losses_w_face['edge'][-1]
                    best_face_state = [latent_w_human.detach().clone(), latent_w_face.detach().clone(), crop_box, j]

            if j >= 100 and j % 50 == 0:
                optim_face = not optim_face
                optim_human = not optim_human
        ################################################################################################################
        # save optimized output
        ################################################################################################################
        latent_w_human, latent_w_face, crop_box, _ = best_face_state 
        latent_final_human = latent_w_human.unsqueeze(0).repeat(G_human.num_ws, 1).unsqueeze(0)   
        with torch.no_grad():
            final_human = G_human.synthesis(latent_final_human, noise_mode='const')   
        
        with torch.no_grad():
            final_face = G_face.synthesis(latent_w_face, noise_mode='const') 
                
        xmin, ymin, xmax, ymax = crop_box
        
        gen_paste = final_human.clone().squeeze()
        gen_paste = 255 * ((gen_paste + 1) / 2)
        gen_paste = gen_paste.cpu().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)

        im = Image.fromarray(gen_paste)
        
        paste_face = final_face.clone().squeeze()
        paste_face = 255 * ((paste_face + 1) / 2)
        paste_face = paste_face.cpu().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        
        paste_im = Image.fromarray(paste_face)
        paste_im = paste_im.resize((ymax-ymin, xmax-xmin), PIL.Image.LANCZOS)
        im.paste(paste_im, (xmin, ymin))
        plt.imshow(im)
        
        im.save(f'{out_folder}/{person}_{body}_optimized.png')
            
        