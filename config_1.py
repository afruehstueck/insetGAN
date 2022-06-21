# Configuration for loosely constrained Face/Body combination

##########################################################################
# Directories
##########################################################################
home_dir = '.'
out_folder = f'{home_dir}/results_config_1'

##########################################################################
# Input parameters for starting canvases and insets
##########################################################################
seed_canvas = 1234
seed_inset = 54321 

# output a selection of seeds to pick a subset of images
output_seed_images = False

# specify selection of bodies and faces
selected_bodies = [3, 9, 3, 9, 3, 9, 6, 11, 6, 11, 6, 11]
selected_faces  = [8, 8, 22, 22, 2, 2, 5, 5, 10, 10, 13, 13] 

# truncation
trunc_canvas = 0.5
#trunc_inset = 0.4 #single truncation value
trunc_insets = [0.8, 0.8, 0.8, 0.7, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3] #layer based truncation (better to get frontally oriented faces while maintaining diversity)

##########################################################################
# Optimization parameters
##########################################################################

learning_rate_optim_canvas = 0.035
learning_rate_optim_inset = 0.006
num_optim_iter = 400

switch_optimizers_every = 40
start_canvas_optim = True #start optimization of canvas first | False = optimize inset first

##########################################################################
# Constraints
##########################################################################

fix_canvas_from_start = False
fix_canvas_at_iter = 75 #-1 for no body constraint

fix_inset_from_start = True
fix_inset_at_iter = -1 

update_bbox_interval = 20
update_bbox_until = 100

edge_loss_increase_until = 300 #slow increase of edge loss influence | 0 for no slow increase

##########################################################################
# Loss combinations
##########################################################################

lambdas_w_inset = {
    'L1': 500, #
    'L1_in': 1000,
    'perceptual_in': 0.4,
    'perceptual': 0.05,  
    'perceptual_edge': 0.25, 
    'edge': 20000, 
}

lambdas_w_canvas = {
    'L1': 1000, 
    'perceptual': 0.15, 
    'edge': 5000,
    'mean_latent': 20000,
    'selected_body': 0.01,
    'selected_body_L1': 1000,
}