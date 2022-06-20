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

G_canvas = networks['G_ema'].to(device)

# define average human
w_samples = G_canvas.mapping(torch.from_numpy(np.random.RandomState(123).randn(10000, G_canvas.z_dim)).to(device), None) 
w_samples = w_samples[:, :1, :]
latent_avg_canvas = torch.mean(w_samples, axis=0).squeeze()
latent_avg_canvas = latent_avg_canvas.unsqueeze(0).repeat(G_canvas.num_ws, 1).unsqueeze(0)   

print('... loaded canvas generator.')

#############################################################################################
# LOAD FACE NETWORK
#############################################################################################
ckpt = f'{home_dir}/networks/ffhq.pkl'

with open(ckpt, 'rb') as f:
    networks = pickle.Unpickler(f).load()   

G_inset = networks['G_ema'].to(device)

# define average face
w_samples = G_inset.mapping(torch.from_numpy(np.random.RandomState(123).randn(10000, G_inset.z_dim)).to(device), None) 
w_samples = w_samples[:, :1, :]
latent_avg_inset = torch.mean(w_samples, axis=0).squeeze()
latent_avg_inset = latent_avg_inset.unsqueeze(0).repeat(G_inset.num_ws, 1).unsqueeze(0)   

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
            disc = D_canvas(optim, None)
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

#### PARAMETRIC CHOICES #### 
seed_canvas = 1111 
seed_inset = 1111 
trunc_canvas = 0.4
trunc_inset = 0.4
trunc_insets = [0.99, 0.9, 0.85, 0.8, 0.7, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

learning_rate_optim_canvas = 0.035
learning_rate_optim_inset = 0.005

num_optim_iter = 400

fix_body_at_iter = 75 #-1 for no body constraint

update_bbox_interval = 25
update_bbox_until = 100

edge_loss_increase_until = 300 #slow increase of edge loss influence | 0 for no slow increase

switch_optimizers_every = 50
start_canvas_optim = True #start optimization of canvas first | False = optimize inset first
##############################

z = torch.from_numpy(np.random.RandomState(seed_canvas).randn(16, G_canvas.z_dim)).to(device)

with torch.no_grad():
    random_humans_w = G_canvas.mapping(z, None) 
    random_humans_w = random_humans_w * (1 - trunc_canvas) + latent_avg_canvas * trunc_canvas
with torch.no_grad():
    random_outputs = G_canvas.synthesis(random_humans_w.to(device), noise_mode='const') 

save_tensor(random_outputs, 'human', out_folder=out_folder)


z = torch.from_numpy(np.random.RandomState(seed_inset).randn(16, G_inset.z_dim)).to(device)
with torch.no_grad():
    random_face_w = G_inset.mapping(z, None) 
    for i in range(18):
        random_face_w[:, i, :] = random_face_w[:, i, :] * (1 - trunc_insets[i]) + latent_avg_inset[:, i, :] * trunc_insets[i]
with torch.no_grad():
    random_outputs = G_inset.synthesis(random_face_w.to(device), noise_mode='const') 

save_tensor(random_outputs, 'face', out_folder=out_folder)


data_transforms = transforms.Compose([
    transforms.ToTensor()
])

plt.figure(figsize=(14, 14))

face_res = G_inset.synthesis.img_resolution 

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

lambdas_w_inset = {
    'L1': 500, #
    'perceptual_in': 0.55,
    'perceptual': 0.05,  
    'perceptual_edge': 0.1, 
    'edge': 40000, 
}

lambdas_w_canvas = {
    'L1': 500, 
    'perceptual': 0.15, 
    'edge': 2500,
    'mean_latent': 25000,
    'selected_body': 0.01,
    'selected_body_L1': 1000,
}

bodies = random_humans_w[[0, 1]]
faces = random_face_w[[0, 1, 3, 4, 5, 8, 12]]

for face in range(len(faces)): 
    for body in range(len(bodies)):
        
        # get respective start latents for face and body 
        latent_w_canvas = bodies[body][0].detach().clone()
        latent_in_canvas = latent_w_canvas.unsqueeze(0).unsqueeze(0).repeat(1, G_canvas.num_ws, 1).to(device)
        
        latent_w_inset = faces[face].unsqueeze(0).detach().clone()

        losses_w_inset = defaultdict(list)
        losses_w_canvas = defaultdict(list)
        
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad():
                gen_canvas = G_canvas.synthesis(latent_in_canvas, noise_mode='const')   
                hires_inset = G_inset.synthesis(latent_w_inset, noise_mode='const') 
            save_image(hires_inset, f'{face}_face_input', out_folder=out_folder)
        
        if face_res == 1024:
            input_inset = downsampler_256(hires_inset) 
        else:
            input_inset = hires_inset

        optim_w = []     

        ctx_frame = 16

        ################################################################################################################
        # find target regions for canvas and insets
        ################################################################################################################        
            
        face_bounding_box = get_bounding_box_face(mtcnn, input_inset)
        input_downsampled = ( downsampler_64(hires_inset) if face_res == 1024 else downsampler(hires_inset) )

        human_bounding_box = get_bounding_box_face(mtcnn, gen_canvas)
        
        gen_inset, crop_box = get_target_bounding_box_face(gen_canvas, human_bounding_box.squeeze(), face_bounding_box.squeeze(), vertical=False)
        xmin, ymin, xmax, ymax = crop_box

        w_inset = xmax - xmin 
        h_inset = ymax - ymin

        latent_w_inset_input = latent_w_inset.clone()
        latent_in = latent_w_canvas.unsqueeze(0).repeat(G_canvas.num_ws, 1).unsqueeze(0)   

        ################################################################################################################
        # set up optimization
        ################################################################################################################
        opt_canvas = torch.optim.Adam([latent_w_canvas], lr=learning_rate_optim_canvas)
        opt_inset = torch.optim.Adam([latent_w_inset], lr=learning_rate_optim_inset)

        optim_canvas, optim_inset = start_canvas_optim, ~start_canvas_optim 

        best_inset_state, best_canvas_state = None, None
        best_canvas_loss, best_inset_loss = 1000000, 1000000

        selected_body = None
        pbar = tqdm(range(num_optim_iter), position=1, leave=True)

        for j in pbar: 
            latent_w_inset.requires_grad = optim_inset
            latent_w_canvas.requires_grad = optim_canvas

            optimizer = opt_inset if optim_inset else opt_canvas
            optimizer.zero_grad()                
            
            if j == fix_body_at_iter:
                selected_body = downsampler_128(gen_canvas).squeeze() 
            
            latent_in = latent_w_canvas.unsqueeze(0).repeat(G_canvas.num_ws, 1).unsqueeze(0) 
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                gen_canvas = G_canvas.synthesis(latent_in, noise_mode='const') 

            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                hires_inset = G_inset.synthesis(latent_w_inset, noise_mode='const')

            if face_res == 1024:
                target_inset = downsampler_256(hires_inset) #256 x 256 face
            else:
                target_inset = hires_inset
            target_downsampled = downsampler(target_inset) #64 x 64 face 
                 
            # update bounding boxes
            if j % update_bbox_interval == 0 and j <= update_bbox_until:
                new_canvas_bounding_box = get_bounding_box_face(mtcnn, gen_canvas)
                xmin_bbox, ymin_bbox, xmax_bbox, ymax_bbox = new_canvas_bounding_box

                # check for some human specific constraints to ensure optimization does not become unruly
                if ( new_canvas_bounding_box is not None # ensure bbox could be detected
                and ymax_bbox - ymin_bbox >= 90 and ymax_bbox - ymin_bbox <= 125  # prevent face from becoming too small or large
                and ymin_bbox > 6 and ymin_bbox < 64  # bbox should stay close to upper edge of image
                and abs(xmin_bbox - 512) < 200 ):  # bbox should stay close to center of image
                    delta_bounding_box = new_canvas_bounding_box - human_bounding_box
                    human_bounding_box = human_bounding_box + ((250-j)/250) * delta_bounding_box # update bbox by decreasing amounts

            gen_inset, crop_box = get_target_bounding_box_face(gen_canvas, human_bounding_box.squeeze(), face_bounding_box.squeeze(), vertical=False)
            xmin, ymin, xmax, ymax = crop_box

            gen_downsampled = downsampler(gen_inset) #64 x 64 human face region

            # accumulate losses
            total_loss = 0        
            loss_dict = losses_w_inset if optim_inset else losses_w_canvas
            loss_source = lambdas_w_inset if optim_inset else lambdas_w_canvas

            loss_info_names = []
            loss_info_quantities = []
            for loss_name in loss_source.keys():  
                loss = 0
                loss_info_names.append(loss_name)
                loss_weight = loss_source[loss_name]

                if 'edge' in loss_name and j <= edge_loss_increase_until: # slowly increase edge loss influence
                    loss_weight = loss_weight * max(j - edge_loss_increase_until, 0) / edge_loss_increase_until

                loss_fn = loss_fn_dict[loss_name]

                if face_bounding_box is not None:
                    xmin_inset, ymin_inset, xmax_inset, ymax_inset = face_bounding_box
                    xmin_inset_down, ymin_inset_down, xmax_inset_down, ymax_inset_down = np.array([xmin_inset, ymin_inset, xmax_inset, ymax_inset], dtype=np.uint8) // 4
                    xdiff = xmax_inset - xmin_inset
                    xmid_inset = int(xmin_inset + float(xdiff / 2))
                    xdiff_mult4 = ((16 * round(xdiff/16))) // 2 + 16
                    xmin_inset_border = max(xmid_inset - xdiff_mult4, 0)
                    xmax_inset_border = min(xmid_inset + xdiff_mult4, 256)
                    ymax_inset = int(ymax_inset)
                if optim_inset:
                    if loss_name == 'perceptual_in': #constrain interior region of image stay close to input face
                        if input_downsampled is not None:
                            loss = loss_fn(input_downsampled[:, :, 2:ymax_inset_down-2, xmin_inset_down+2:xmax_inset_down-2], target_downsampled[:, :, 2:ymax_inset_down-2, xmin_inset_down+2:xmax_inset_down-2])
                        else:
                            loss = torch.zeros_like(total_loss).to(device)
                    else:
                        if loss_name in loss_fn_downsample:
                            t, g = target_downsampled, gen_downsampled 
                        else:
                            t, g = target_inset, gen_inset
                        loss = loss_fn(t, g)
                else: # optim_canvas
                    if loss_name == 'mean_latent':
                        loss = loss_fn(latent_in, latent_avg_canvas.clone())    
                    elif 'selected_body' in loss_name:
                        if selected_body is not None:
                            downsampled_body = downsampler_128(gen_canvas)
                            t = V(selected_body[:, 28:, 32:-32])
                            g = downsampled_body[:, :, 28:, 32:-32]
                            loss = loss_fn(t, g)   
                        else:
                            loss = torch.zeros_like(total_loss).to(device)
                    else:
                        t = target_downsampled if loss_name in loss_fn_downsample else target_inset
                        g = gen_downsampled if loss_name in loss_fn_downsample else gen_inset

                        loss = loss_fn(t, g)
                total_loss += loss_weight * loss

                loss_val = loss.clone().detach().cpu().numpy()
                loss_info_quantities.append(loss_weight * loss_val)
                loss_dict[loss_name].append(loss_val)

            total_loss.backward()
            optimizer.step()

            if optim_inset:
                if (j > 0 and losses_w_inset['edge'][-1] < best_inset_loss and losses_w_canvas['edge'][-1] <= best_canvas_loss if 'edge ' in losses_w_canvas else True):
                    best_inset_loss = losses_w_inset['edge'][-1]
                    best_inset_state = [latent_w_canvas.detach().clone(), latent_w_inset.detach().clone(), crop_box, j]

            # switch optimizers 

            if j % switch_optimizers_every == 0:
                optim_inset = not optim_inset
                optim_canvas = not optim_canvas

        ################################################################################################################
        # save optimized output
        ################################################################################################################
        latent_w_canvas, latent_w_inset, crop_box, _ = best_inset_state 
        latent_final_canvas = latent_w_canvas.unsqueeze(0).repeat(G_canvas.num_ws, 1).unsqueeze(0)   
        with torch.no_grad():
            final_canvas = G_canvas.synthesis(latent_final_canvas, noise_mode='const')   
        
        with torch.no_grad():
            final_inset = G_inset.synthesis(latent_w_inset, noise_mode='const') 
                
        xmin, ymin, xmax, ymax = crop_box
        
        gen_paste = final_canvas.clone().squeeze()
        gen_paste = 255 * ((gen_paste + 1) / 2)
        gen_paste = gen_paste.cpu().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)

        im = Image.fromarray(gen_paste)
        
        paste_inset = final_inset.clone().squeeze()
        paste_inset = 255 * ((paste_inset + 1) / 2)
        paste_inset = paste_inset.cpu().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        
        paste_im = Image.fromarray(paste_inset)
        paste_im = paste_im.resize((ymax-ymin, xmax-xmin), PIL.Image.LANCZOS)
        im.paste(paste_im, (xmin, ymin))
        plt.imshow(im)
        
        im.save(f'{out_folder}/{face}_{body}_optimized.png')
            
        