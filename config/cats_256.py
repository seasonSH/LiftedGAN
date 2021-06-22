''' Config Proto '''

####### INPUT OUTPUT #######

# The name of the project folder
name = 'default'

# The folder to save log and model
log_base_dir = './log_cats_256/'

# The interval between display in terminal
print_interval = 10

# The interval between writing summary
summary_interval = 500

# Number of samples to generate for validation
n_samples = 32

# The path to the pretrained StyleGAN2 checkpoint
generator_path = './pretrained/stylegan2-cats-cropped-300000.pt'

# The path to the pretrained face recognition embedding
face_embedding_path = None

# The resolution of generator
image_size = 256

# Number of workers for data loading
num_workers = 4

# Base random seed for data fetching
base_random_seed = 1



####### TRAINING STRATEGY #######

# Number of samples in a batch
batch_size = 8

# Number of epochs
num_epochs = 10

# Number of steps per epoch
epoch_size = 5000

# learning rate
learning_rate = 1e-4

# The path to the checkpoint
restore_model = None


## Model and Loss
truncation = 0.7
min_depth = 0.9
max_depth = 1.1
border_depth = 1.05
xyz_rotation_range = 60  # (-r,r) in degrees
xy_translation_range = 0.1  # (-t,t) in 3D
z_translation_range = 0.01  # (-t,t) in 3D
n_scale = 3
vgg_indices = [1,2,3]
use_flip = True
lam_rec = 5.0
lam_perc = 1.0
lam_flip = 0.8
lam_perturb_im = 2.0
lam_perturb_param = 4.0
lam_identity_perturb = 0.0
lam_reg_style = 0.1
lam_albedo_rank = 5e-3
rank_downsample = 4
perc_type = 'cosine'
clip_render_grad = True
clamp_border = True
generator_texture = True

renderer_params = {
    'device': 'cuda',
    'image_size': 256,
    'rot_center_depth': 1.00,
    'fov': 10, # in degrees
    'tex_cube_size': 2,
    'min_depth': min_depth,
    'max_depth': max_depth,
    'min_rho': 0.5,
    'max_rho': 1.0,
    'min_phi': -90,
    'max_phi': 90,
    'scale_x': 0.18,
    'scale_y': 0.12,
    'scale_z': 0.12,
    'scale_xy': 1.0,
    'border_depth': border_depth,
}

blur_params = {
    'size': 10,
    'sigma': 4
}
