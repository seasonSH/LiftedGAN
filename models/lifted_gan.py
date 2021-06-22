import os
import sys
import imp
import math
import time
import glob
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from models.renderers.renderer import Renderer
from models.networks_stylegan2 import Generator
from models.face_embedding import LResNet
from models.perc_loss import PerceptualLoss
from models import networks

class LiftedGAN(object):
    def __init__(self):
        return

    def initialize(self, config, training=True):
        self.global_step = 0
        self.set_config(config)

        self.netD = networks.DepthNet(in_dim=512, out_dim=1, z_dim=512, n_mlp=4, activation=None, bias=True).cuda()
        self.netT = networks.TransformationNet(in_dim=512, out_dim=1, z_dim=16, n_mlp=4, nf=16, activation=None).cuda()
        self.netSD = networks.StyleDecomposeNet(style_dim=512, n_layer=4).cuda()
        self.netSC = networks.StyleComposeNet(style_dim=512, n_layer=4).cuda()
        self.blur = networks.Blur(**self.config.blur_params)
        self.ContourGradientClip = networks.ContourGradientClip
        self.PerceptualLoss = PerceptualLoss(requires_grad=False,
            loss_type=self.config.perc_type, n_scale=self.config.n_scale, slice_indices=self.config.vgg_indices).cuda()

        print(f"Loading generator from: {self.config.generator_path}")
        self.generator = Generator(self.config.image_size, 512, 8).cuda()
        self.generator.load_state_dict(torch.load(self.config.generator_path)['g_ema'], strict=False)
        self.generator.eval()
        for p in self.generator.parameters():
            p.requires_grad = False
        self.get_statistics()

        if self.config.face_embedding_path is not None:
            self.FaceEmbedding = LResNet(nch=3, image_size=(128,128), nlayer=18)
            self.FaceEmbedding = torch.nn.DataParallel(self.FaceEmbedding).cuda()
            self.FaceEmbedding.load_state_dict(torch.load(self.config.face_embedding_path)['model_state_dict'])
            self.FaceEmbedding.eval()

        self.renderer = Renderer(**config.renderer_params)

        self.network_names = [k for k in vars(self) if 'net' in k]

        if training:
            optimize_modules = [getattr(self, net_name) for net_name in self.network_names]
            optimize_parameters = list(itertools.chain.from_iterable([m.parameters() for m in optimize_modules]))
            self.optimizer = torch.optim.Adam(optimize_parameters, lr=config.learning_rate, betas=(0.9,0.999), weight_decay=5e-4)


    def set_config(self, config):
        self.max_depth = config.max_depth
        self.min_depth = config.min_depth
        self.border_depth = config.border_depth
        self.xyz_rotation_range = config.xyz_rotation_range
        self.xy_translation_range = config.xy_translation_range
        self.z_translation_range = config.z_translation_range
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.depth_inv_rescaler = lambda d :  (d-self.min_depth) / (self.max_depth-self.min_depth)  # (min_depth,max_depth) => (0,1)
        self.config = config

    def restore_model(self, model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        for net in self.network_names:
            getattr(self, net).load_state_dict(checkpoint[net])
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('WARNING: failed to restore optimizer')
        self.global_step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        del checkpoint
        return epoch

    def load_model(self, model_path, initialize=True, strict=False):
        if initialize:
            model_dir = os.path.dirname(os.path.abspath(model_path))
            config_file = os.path.join(model_dir, 'config.py')
            config = imp.load_source('config_file_', config_file)
            self.initialize(config)
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        for net in self.network_names:
            if (not net in checkpoint) and not strict: continue
            getattr(self, net).load_state_dict(checkpoint[net])

    def save_model(self, save_path, epoch, max_checkpoints=5):
        filename = os.path.join(save_path, f'checkpoint_{epoch:03}.pth')
        states = {}
        for net in self.network_names:
            states[net] = getattr(self, net).state_dict()
        states['optimizer'] = self.optimizer.state_dict()
        states['global_step'] = self.global_step
        states['epoch'] = epoch
        print('Saving to checkpoint: {}'.format(filename))
        torch.save(states, filename)

        # Remove the old checkpoints
        existing_files = glob.glob('{}/checkpoint_*'.format(save_path))
        timestamps = [int((f.split('checkpoint_')[1].split('.pth')[0])) for f in existing_files]
        sorted_indices = np.argsort(timestamps)
        for idx in sorted_indices[:-max_checkpoints]:
            os.remove(existing_files[idx])

    def get_statistics(self):
        with torch.no_grad():
            code = torch.randn(50000, 512).cuda()
            styles = self.generator.style(code)
            self.w_mu = styles.mean(0, keepdim=True)
            self.w_sigma_inv = 0.5 / styles.std(0).mean().pow(2)
            print(f'w_sigma_inv: {self.w_sigma_inv}')


    def photometric_loss(self, im1, im2, mask=None):
        loss = (im1-im2).abs()
        if mask is None:
            loss = loss.mean()
        else:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        return loss

    def image_loss(self, recon_im, input_im, lam_perc, mask=None):
        loss = self.photometric_loss(recon_im, input_im, mask=mask)
        if lam_perc > 0:
            loss_perc = self.PerceptualLoss(recon_im, input_im, mask=mask).mean()
            loss = loss + lam_perc * loss_perc
        return loss

    def symmetric_image_loss(self, recon_im, recon_im_flip, input_im, lam_perc, lam_flip, mask=None):

        if isinstance(mask, tuple):
            mask, mask_flip = mask
        else:
            mask_flip = mask

        # Photometric Loss
        loss_l1_im = self.photometric_loss(recon_im, input_im, mask=mask)
        loss_rec = loss_l1_im

        # Perceptual Loss
        if lam_perc > 0:
            loss_perc_im = self.PerceptualLoss(recon_im, input_im, mask=mask)
            loss_rec += lam_perc * loss_perc_im.mean()

        # Photometric Loss (flip)
        if lam_flip > 0:
            loss_l1_im_flip = self.photometric_loss(recon_im_flip, input_im, mask=mask_flip)
            loss_rec += lam_flip * loss_l1_im_flip

        # Perceptual Loss (flip)
        if lam_perc > 0 and lam_flip > 0:
            loss_perc_im_flip = self.PerceptualLoss(recon_im_flip, input_im, mask=mask_flip)
            loss_rec += lam_flip * lam_perc * loss_perc_im_flip.mean()

        return loss_rec



    def estimate(self, input_style):
        b = input_style.shape[0]
        size = self.config.image_size

        ## decompose the style code into neutralized style, light and view
        neutral_style, canon_light, view = self.netSD(input_style)
        neutral_light = torch.stack([
            canon_light[:,0],
            canon_light[:,1],
            canon_light[:,2]*0.0,
            canon_light[:,3]*0.0,
            ], 1)
        neutral_style = self.netSC(input_style.detach(), neutral_light, 0*view)

        ## predict canonical albedo
        canon_albedo, style_feat = self.generator(neutral_style,
                input_is_latent=True, randomize_noise=False, return_feat=True)
        canon_im_raw = canon_albedo

        ## predict canonical depth
        canon_depth_raw = self.netD(neutral_style, style_feat)  # BxHxW
        canon_depth_raw = canon_depth_raw.squeeze(1)
        canon_depth = canon_depth_raw - canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
        canon_depth = canon_depth.tanh()
        canon_depth = self.depth_rescaler(canon_depth)

        # clamp border depth
        if self.config.clamp_border:
            _, h, w = canon_depth.shape
            border_with = 2
            depth_border = torch.zeros(1,h,w-2*border_with).to(canon_depth.device)
            depth_border = F.pad(depth_border, (border_with,border_with), mode='constant', value=1)
            canon_depth = canon_depth*(1-depth_border) + depth_border*self.border_depth


        trans_map = self.netT(neutral_style)
        trans_map = torch.sigmoid(trans_map+5)

        depth_mask = (canon_depth[:,None] <= (self.border_depth-0.01)).float()
        trans_map = depth_mask * torch.ones_like(trans_map) + (1-depth_mask) * trans_map
        trans_map = self.blur(trans_map)
        trans_map = F.pad(trans_map[:,:,1:-1,2:-2], (2,2,1,1), mode='constant', value=0.0)


        # Delight
        if self.config.generator_texture:
            canon_light_a, canon_light_b, canon_light_d = self.parse_light(canon_light, frontalize=True)
            canon_normal = self.renderer.get_normal_from_depth(canon_depth)
            canon_diffuse_shading = (canon_normal * canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
            canon_shading = canon_light_a.view(-1,1,1,1) + canon_light_b.view(-1,1,1,1)*canon_diffuse_shading
            canon_albedo = (canon_albedo/2+0.5) / (canon_shading+1e-8) *2-1

        return canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw


    def parse_light(self, canon_light, frontalize=False):
        b = canon_light.size(0)
        canon_light_a = canon_light[:,:1] /2+0.5  # ambience term
        canon_light_b = canon_light[:,1:2] /2+0.5  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        if frontalize:
            canon_light_d = torch.cat([0*canon_light_dxy, torch.ones(b,1)], 1)
        else:
            canon_light_d = torch.cat([canon_light_dxy, torch.ones(b,1)], 1)
        canon_light_d = canon_light_d / ((canon_light_d**2).sum(1, keepdim=True))**0.5  # diffuse light direction
        return canon_light_a, canon_light_b, canon_light_d


    def render(self, canon_depth, canon_albedo, canon_light, view, use_light=True, trans_map=None):
        b, c, h, w = canon_albedo.shape

        ## predict viewpoint transformation
        view = torch.cat([
            view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            view[:,3:5] *self.xy_translation_range,
            view[:,5:] *self.z_translation_range], 1)

        ## predict lighting
        canon_light_a, canon_light_b, canon_light_d = self.parse_light(canon_light)

        ## shading
        canon_normal = self.renderer.get_normal_from_depth(canon_depth)
        canon_diffuse_shading = (canon_normal * canon_light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a.view(-1,1,1,1) + canon_light_b.view(-1,1,1,1)*canon_diffuse_shading
        if use_light:
            canon_im = (canon_albedo/2+0.5) * canon_shading *2-1
        else:
            canon_im = canon_albedo

        ## reconstruct input view
        self.renderer.set_transform_matrices(view)
        canon_depth_ = self.ContourGradientClip.apply(canon_depth) if self.config.clip_render_grad else canon_depth
        recon_im, recon_depth, silhouette =  self.renderer.render(
            canon_depth_, canon_im, mask=trans_map, get_depth=True, get_alpha=True)
        silhouette = 2*silhouette.unsqueeze(1)
        silhouette = (silhouette > 0.5).float()

        return recon_im, canon_im, canon_normal, canon_shading, silhouette, recon_depth.detach()


    def perturb(self, neutral_style, light, view):
        b = neutral_style.size(0)

        # Light
        indices_light_1 = torch.randperm(light.size(0))
        indices_light_2 = torch.randperm(light.size(0))
        perturbed_light = torch.cat((light[indices_light_1,:2],light[indices_light_2,2:]), 1)

        # Random Angle
        amin = torch.tensor([[-15,-45,0]]).float()
        amax = torch.tensor([[15,45,0]]).float()
        angles = torch.rand(b,3)
        angles = (angles*(amax-amin) + amin) #*  min(1.0, 0.5+0.5*self.global_step/4000)
        trans_x = torch.sin(angles[:,1:2]/180*math.pi) * 0.2
        perturbed_view = torch.cat([
            angles[:,:3] / self.xyz_rotation_range, trans_x, 0*view[:,4:]], 1)

        return neutral_style, perturbed_light.detach(), perturbed_view.detach(), angles


    def train_step(self):

        watchlist = {}
        summary = {'scalar':{}, 'histogram':{}, 'image':{}}
        b0 = min(12,self.config.batch_size)
        nrow = int(math.ceil(math.sqrt(b0)))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # zero gradient
        for net in self.network_names:
            getattr(self, net).train()

        ### Sample from generator
        with torch.no_grad():
            code = torch.randn(self.config.batch_size, 512)
            styles = self.generator.style(code)
            styles = self.config.truncation * styles + (1-self.config.truncation) * self.w_mu
            input_im = self.generator(styles, input_is_latent=True, randomize_noise=False)
            input_im = input_im.clamp(min=-1,max=1).contiguous()
        summary['image']['input/input_im'] = make_grid(input_im[:b0].clamp(min=-1,max=1)*0.5+0.5, nrow=nrow, normalize=False)

        ### Estimation
        canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = self.estimate(styles)
        summary['image']['depth/canon_depth'] = make_grid(self.depth_inv_rescaler(canon_depth[:,None])[:b0], nrow=nrow, normalize=True)
        summary['image']['image/canon_im_raw'] = make_grid(canon_im_raw[:b0], nrow=nrow, normalize=True)
        summary['image']['image/canon_albedo'] = make_grid(canon_albedo[:b0], nrow=nrow, normalize=True)
        summary['image']['image/trans_map'] = make_grid(trans_map[:b0], nrow=nrow, normalize=False)
        summary['histogram']['view/rotation_x'] = view[:,0] / math.pi * 180
        summary['histogram']['view/rotation_y'] = view[:,1] / math.pi * 180
        summary['histogram']['view/rotation_z'] = view[:,2] / math.pi * 180
        summary['histogram']['view/translation_x'] = view[:,3]
        summary['histogram']['view/translation_y'] = view[:,4]
        summary['histogram']['view/translation_z'] = view[:,5]
        summary['histogram']['light/ambient'] = canon_light[:,0]
        summary['histogram']['light/diffuse'] = canon_light[:,1]
        summary['histogram']['light/dx'] = canon_light[:,2]
        summary['histogram']['light/dy'] = canon_light[:,3]
        self.canon_depth = canon_depth


        ### Rendering
        recon_im, canon_im, canon_normal, canon_shading, silhouette, recon_depth = \
                self.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)
        recon_im_flip, canon_im_flip, _, canon_shading_flip, silhouette_flip, recon_depth_flip = \
                self.render(canon_depth.flip(2), canon_albedo.flip(3), canon_light, view, trans_map=trans_map)

        summary['image']['depth/recon_depth'] = make_grid(recon_depth[:b0].unsqueeze(1), nrow=nrow, normalize=True)
        summary['image']['depth/canon_normal'] = make_grid(canon_normal[:b0].permute(0,3,1,2), nrow=nrow, normalize=True)
        summary['image']['depth/canon_shading'] = make_grid(canon_shading[:b0], nrow=nrow, normalize=True)
        summary['image']['image/canon_im'] = make_grid(canon_im[:b0], nrow=nrow, normalize=True)
        summary['image']['image/recon_im'] = make_grid(recon_im[:b0], nrow=nrow, normalize=True)
        summary['image']['depth/canon_shading_flip'] = make_grid(canon_shading_flip[:b0], nrow=nrow, normalize=True)
        summary['image']['image/canon_im_flip'] = make_grid(canon_im_flip[:b0], nrow=nrow, normalize=True)
        summary['image']['image/recon_im_flip'] = make_grid(recon_im_flip[:b0], nrow=nrow, normalize=True)
        self.canon_normal = canon_normal
        self.canon_im = canon_im


        ### Rendering images from perturbed parameters
        _, perturbed_light_original, perturbed_view_original, perturb_angles = self.perturb(neutral_style, canon_light, view)
        perturbed_styles = self.netSC(styles.detach(), perturbed_light_original, perturbed_view_original)
        _, perturbed_light, perturbed_view = self.netSD(perturbed_styles)

        perturbed_im = self.render(canon_depth, canon_albedo, perturbed_light, perturbed_view, trans_map=trans_map)[0]
        summary['image']['perturb/perturbed_im'] = make_grid(perturbed_im[:b0].clamp(min=-1,max=1)*0.5+0.5, nrow=nrow, normalize=False)

        ### Compute Losses
        loss_total = torch.tensor(0.).cuda()

        # Symmetric Image Reconstruction Loss
        if self.config.lam_rec > 0:
            loss_rec = self.symmetric_image_loss(recon_im, recon_im_flip, input_im,
                lam_perc=self.config.lam_perc, lam_flip=self.config.lam_flip)
            loss_total += self.config.lam_rec * loss_rec
            watchlist['loss_rec'] = loss_rec.item()

        # Part I of Image perturbation loss: make perturbed img resemble generated img
        if self.config.lam_perturb_im > 0:
            with torch.no_grad():
                regenerated_im = self.generator(perturbed_styles, input_is_latent=True, randomize_noise=False)
                summary['image']['perturb/regenerated_im'] = make_grid(regenerated_im[:b0].clamp(min=-1,max=1)*0.5+0.5, nrow=nrow, normalize=False)
            loss_perturb_im = self.image_loss(perturbed_im, regenerated_im.detach(), lam_perc=self.config.lam_perc)
            loss_total += self.config.lam_perturb_im * loss_perturb_im
            watchlist['loss_perturb_im'] = loss_perturb_im.item()

        # Regularization over re-generated parameters
        if self.config.lam_perturb_param > 0:
            loss_perturb_light = (perturbed_light - perturbed_light_original).pow(2).mean()
            loss_perturb_view = (perturbed_view - perturbed_view_original)[:,:3].pow(2).mean()
            loss_total += self.config.lam_perturb_param * (loss_perturb_light + loss_perturb_view)
            watchlist['loss_perturb_light'] = loss_perturb_light.item()
            watchlist['loss_perturb_view'] = loss_perturb_view.item()

        # Identity Regularization over Pertubed Images
        if self.config.lam_identity_perturb > 0:
            indices = (perturb_angles[:,1] > -25) & (perturb_angles[:,1] < 25)
            embed_canon = self.FaceEmbedding(canon_im_raw)[indices]
            embed_perturb = self.FaceEmbedding(perturbed_im)[indices]
            loss_ID_perturb = (embed_canon - embed_perturb).pow(2).sum(1).mean()
            loss_total += self.config.lam_identity_perturb * loss_ID_perturb
            watchlist['loss_ID_perturb'] = loss_ID_perturb.item()

        # Latent Regularization loss
        if self.config.lam_reg_style > 0:
            loss_reg_style = (perturbed_styles - self.w_mu).pow(2).mean() * self.w_sigma_inv
            loss_total += self.config.lam_reg_style * loss_reg_style
            watchlist['loss_reg_style'] = loss_reg_style.item()

        # Low Rank Regularization
        mask = (canon_depth[:,None] <= (self.border_depth-0.01)).float()
        if self.config.lam_albedo_rank > 0:
            albedo = canon_albedo.mean(1, keepdim=True)
            albedo_residual = albedo - self.blur(albedo)
            albedo_residual = albedo_residual * mask
            summary['image']['other/albedo_residual'] = make_grid(albedo_residual[:b0], nrow=nrow, normalize=True)
            if self.config.rank_downsample:
                albedo_residual = F.avg_pool2d(albedo_residual, self.config.rank_downsample, stride=self.config.rank_downsample, padding=0)
            albedo_residual_flat = albedo_residual.reshape(albedo_residual.size(0), -1)
            loss_albedo_rank = torch.norm(albedo_residual_flat, p='nuc')
            if self.config.lam_albedo_rank > 0:
                loss_total += self.config.lam_albedo_rank * loss_albedo_rank
                watchlist['loss_albedo_rank'] = loss_albedo_rank.item()


        self.optimizer.zero_grad()
        watchlist['loss_total'] = loss_total.item()
        loss_total.backward()

        # Part II of Image perturbation loss: make generated img resemble perturbed img
        # We calculate the gradient separately here to reduce memory cost
        if self.config.lam_perturb_im > 0:
            with torch.no_grad():
                perturbed_im_original = self.render(canon_depth, canon_albedo, perturbed_light_original, perturbed_view_original, trans_map=trans_map)[0]
                summary['image']['perturb/perturbed_im_original'] = make_grid(perturbed_im_original[:b0].clamp(min=-1,max=1)*0.5+0.5, nrow=nrow, normalize=False)

            perturbed_styles = self.netSC(styles.detach(), perturbed_light_original, perturbed_view_original)
            regenerated_im = self.generator(perturbed_styles, input_is_latent=True, randomize_noise=False)
            loss_perturb_G = self.image_loss(perturbed_im_original.detach(), regenerated_im, lam_perc=self.config.lam_perc, mask=None)
            watchlist['loss_perturb_G'] = loss_perturb_G.item()
            (self.config.lam_perturb_im * loss_perturb_G).backward()

        self.optimizer.step()
        self.global_step += 1

        # Collect the summary
        for k,v in watchlist.items():
            if type(v) in [float, int, bool]:
                summary['scalar']['train/'+k] = v

        torch.set_default_tensor_type('torch.FloatTensor')
        return watchlist, summary, self.global_step


    def get_recon_normal(self, canon_normal, canon_depth, view):
        view = torch.cat([
            view[:,:3] *math.pi/180 *self.xyz_rotation_range,
            view[:,3:5] *self.xy_translation_range,
            view[:,5:] *self.z_translation_range], 1)
        self.renderer.set_transform_matrices(view)
        canon_normal = canon_normal.permute(0,3,1,2)
        recon_normal =  self.renderer.render(
            canon_depth, canon_normal, mask=None, get_depth=False)
        return recon_normal

    def get_neutral_shading(self, canon_normal):
        light_d = torch.tensor([[0,0,1]]).repeat(canon_normal.size(0),1).to(canon_normal.device)
        diffuse_shading = (canon_normal * light_d.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        neutral_shading = (0 + 0.8*diffuse_shading).repeat(1,3,1,1).permute(0,2,3,1) *2 - 1
        return neutral_shading


    def add_videos(self, summary):
        b0 = min(12,self.config.batch_size)
        nrow = int(math.ceil(math.sqrt(b0)))
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.canon_im.device).repeat(b0,1)
            neutral_shading = self.get_neutral_shading(self.canon_normal)
            canon_im_rotate = self.renderer.render_yaw(self.canon_im[:b0], self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu()/2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate = self.renderer.render_yaw(self.canon_normal[:b0].permute(0,3,1,2), self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu()/2.+0.5  # (B,T,C,H,W)
            neutral_shading_rotate = self.renderer.render_yaw(neutral_shading[:b0].permute(0,3,1,2), self.canon_depth[:b0], v_before=v0, maxr=90).detach().cpu()/2.+0.5  # (B,T,C,H,W)

            canon_im_rotate_grid = [make_grid(img, nrow=nrow) for img in torch.unbind(canon_im_rotate, 1)]  # [(C,H,W)]*T
            canon_im_rotate_grid = torch.stack(canon_im_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
            canon_normal_rotate_grid = [make_grid(img, nrow=nrow) for img in torch.unbind(canon_normal_rotate, 1)]  # [(C,H,W)]*T
            canon_normal_rotate_grid = torch.stack(canon_normal_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)
            neutral_shading_rotate_grid = [make_grid(img, nrow=nrow) for img in torch.unbind(neutral_shading_rotate, 1)]  # [(C,H,W)]*T
            neutral_shading_rotate_grid = torch.stack(neutral_shading_rotate_grid, 0).unsqueeze(0)  # (1,T,C,H,W)

        summary['image']['image/recon_im_side'] = make_grid(canon_im_rotate[:,0,:,:,:], nrow=nrow, normalize=True)
        if not 'video' in summary: summary['video'] = {}
        summary['video']['rotate/canon_im'] = canon_im_rotate_grid
        summary['video']['rotate/canon_normal'] = canon_normal_rotate_grid
        summary['video']['rotate/neutral_shading'] = neutral_shading_rotate_grid

        return summary


    def test(self, input_code, render=False, recon_normal=False, generate=False, truncation=1.0):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        for net in self.network_names:
            getattr(self, net).eval()
        with torch.no_grad():
            input_batch = input_code.cuda()
            input_style = self.generator.style(input_batch)
            if truncation < 1:
                input_style = truncation * input_style + (1-truncation) * self.w_mu
            canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = self.estimate(input_style)
            keys = ['canon_depth', 'canon_albedo', 'canon_light', 'view', 'neutral_style']

            if generate:
                image_size = self.config.image_size
                gen_im = self.generator(input_style, input_is_latent=True, randomize_noise=False)
                input_im = gen_im
                keys += ['gen_im']
            else:
                input_im = input_batch

            if render:
                recon_im, canon_im, canon_normal, canon_shading, silhouette, recon_depth = \
                    self.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)
                keys += ['recon_im', 'canon_im', 'canon_normal', 'canon_shading', 'silhouette', 'recon_depth']

            if recon_normal:
                recon_normal = self.get_recon_normal(canon_normal, canon_depth, view)
                neutral_shading = self.get_neutral_shading(canon_normal)
                keys += ['recon_normal', 'neutral_shading']

            view = torch.cat([
                view[:,:3] *math.pi/180 *self.xyz_rotation_range,
                view[:,3:5] *self.xy_translation_range,
                view[:,5:] *self.z_translation_range], 1)


            # Collect variables
            local_variables = locals()
            results = {k:local_variables[k].cpu() for k in keys}

        torch.set_default_tensor_type('torch.FloatTensor')
        return results
