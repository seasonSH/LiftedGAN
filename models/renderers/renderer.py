''' Modified from Wu et al. 2020 to Pytorch3D renderer '''
import torch
import math
from .utils import *

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_rotation,
    OpenGLPerspectiveCameras,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.materials import Materials

EPS = 1e-7


class Renderer(torch.nn.Module):
    def __init__(self, **cfgs):
        super().__init__()
        self.device = cfgs.get('device', 'cuda')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.rot_center_depth = cfgs.get('rot_center_depth', (self.min_depth+self.max_depth)/2)
        self.border_depth = cfgs.get('border_depth', 0.3*self.min_depth+0.7*self.max_depth)
        self.fov = cfgs.get('fov', 10)

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        ## renderer for visualization
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = torch.FloatTensor(R).to(self.device)
        t = torch.zeros(1,3, dtype=torch.float32).to(self.device)
        fx = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        fy = (self.image_size-1)/2/(math.tan(self.fov/2 *math.pi/180))
        cx = (self.image_size-1)/2
        cy = (self.image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)


        # Initialize an OpenGL perspective camera.
        R = look_at_rotation(((0,0,0),), at=((0, 0, 1),), up=((0, -1, 0),))
        cameras = OpenGLPerspectiveCameras(device=self.device, fov=self.fov, R=R)
        lights = DirectionalLights(
            ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0), ), direction=((0, 1, 0), ),
            device=self.device,)
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer_torch=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

    def set_transform_matrices(self, view):
        self.rot_mat, self.trans_xyz = get_transform_matrices(view)

    def rotate_pts(self, pts, rot_mat):
        centroid = torch.FloatTensor([0.,0.,self.rot_center_depth]).to(pts.device).view(1,1,3)
        pts = pts - centroid  # move to centroid
        pts = pts.matmul(rot_mat.transpose(2,1))  # rotate
        pts = pts + centroid  # move back
        return pts

    def translate_pts(self, pts, trans_xyz):
        return pts + trans_xyz

    # Original Implementation
    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d

    def clamp_border(self, depth):
        ## clamp border depth
        depth = depth[:,:,:,0]
        _, h, w = depth.shape
        border_width = 2
        depth_border = torch.zeros(1,h,w-2*border_width).to(depth.device)
        depth_border = torch.nn.functional.pad(depth_border, (border_width,border_width), mode='constant', value=1)
        depth = depth*(1-depth_border) + depth_border*self.border_depth
        return depth.unsqueeze(3)

    def grid_3d_to_2d(self, grid_3d):
        b, h, w, _ = grid_3d.shape
        grid_2d = grid_3d / grid_3d[...,2:]
        grid_2d = grid_2d.matmul(self.K.to(grid_3d.device).transpose(2,1))[:,:,:,:2]
        WH = torch.FloatTensor([w-1, h-1]).to(grid_3d.device).view(1,1,1,2)
        grid_2d = grid_2d / WH *2.-1.  # normalize to -1~1
        return grid_2d

    def get_warped_3d_grid(self, depth):
        b, (h, w) = depth.shape[0], depth.shape[-2:]
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat)
        grid_3d = self.translate_pts(grid_3d, self.trans_xyz)
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_inv_warped_3d_grid(self, depth):
        b, (h, w) = depth.shape[0], depth.shape[-2:]
        grid_3d = self.depth_to_3d_grid(depth).reshape(b,-1,3)
        grid_3d = self.translate_pts(grid_3d, -self.trans_xyz)
        grid_3d = self.rotate_pts(grid_3d, self.rot_mat.transpose(2,1))
        return grid_3d.reshape(b,h,w,3) # return 3d vertices

    def get_warped_2d_grid(self, depth):
        b, (h, w) = depth.shape[0], depth.shape[-2:]
        grid_3d = self.get_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d

    def get_inv_warped_2d_grid(self, depth):
        b, (h, w) = depth.shape[0], depth.shape[-2:]
        grid_3d = self.get_inv_warped_3d_grid(depth)
        grid_2d = self.grid_3d_to_2d(grid_3d)
        return grid_2d


    def get_normal_from_depth(self, depth, normalize=True):
        b, (h, w) = depth.shape[0], depth.shape[-2:]
        grid_3d = self.depth_to_3d_grid(depth)

        center = grid_3d[:,1:-1,1:-1]
        left, right, up, down = grid_3d[:,:-2,1:-1], grid_3d[:,2:,1:-1], grid_3d[:,1:-1,:-2], grid_3d[:,1:-1,2:]
        norm1 = torch.cross(up-center,left-center,dim=3)
        norm2 = torch.cross(left-center,down-center,dim=3)
        norm3 = torch.cross(down-center,right-center,dim=3)
        norm4 = torch.cross(right-center,up-center,dim=3)
        normal = norm1 + norm2 + norm3 + norm4

        # Zero Padding
        zero = torch.FloatTensor([0,0,1]).to(depth.device)
        normal = torch.nn.functional.pad(normal.permute(0,3,1,2), (1,1,1,1), mode='replicate').permute(0,2,3,1)
        if normalize:
            normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal


    def render(self, canon_depth, canon_im, mask=None, bcg_color=(1.0,1.0,1.0,), get_depth=False, get_alpha=False):
        b, (h, w) = canon_depth.shape[0], canon_depth.shape[-2:]
        grid_3d_canon = self.depth_to_3d_grid(canon_depth)
        grid_3d = self.get_warped_3d_grid(canon_depth)
        if mask is None:
            mask = torch.ones_like(grid_3d)
        else:
            mask = mask.permute(0,2,3,1)
        grid_3d = mask * grid_3d + (1-mask) * grid_3d_canon
        verts = grid_3d.reshape(b,-1,3)
        faces = get_face_idx(b, h, w).to(canon_depth.device).long()
        rgb = canon_im.permute(0,2,3,1).view(b,-1,3)
        return self.render_torch(verts, faces, rgb, bcg_color, get_depth, get_alpha)


    def render_torch(self, verts, faces, rgb, bcg_color=(1.,1.,1.), get_depth=False, get_alpha=False):
        # b, h, w = grid_3d.shape[:3]
        b = verts.size(0)
        textures = TexturesVertex(verts_features=rgb.view(b,-1,3))
        mesh = Meshes(verts=verts, faces=faces, textures=textures)

        fragments = self.rasterizer_torch(mesh)
        texels = mesh.sample_textures(fragments)
        materials = Materials(device=verts.device)
        blend_params = BlendParams(background_color=bcg_color)
        images = hard_rgb_blend(texels, fragments, blend_params)
        images = images[...,:3].permute(0,3,1,2)

        out = (images,)
        if get_depth:
            depth = fragments.zbuf[...,0]
            mask = (depth==-1.0).float()
            max_depth = self.max_depth + 0.5*(self.max_depth-self.min_depth)
            depth =  mask * max_depth * torch.ones_like(depth) + (1-mask) * depth
            out = out + (depth,)
        if get_alpha:
            colors = torch.ones_like(fragments.bary_coords)
            blend_params = BlendParams(sigma=1e-2, gamma=1e-4, background_color=(1.,1.,1.))
            alpha = sigmoid_alpha_blend(colors, fragments, blend_params)[...,-1]
            out = tuple(out) + (alpha,)
        if len(out) == 1:
            out = out[0]
        return out


    def render_yaw(self, im, depth, v_before=None, v_after=None, rotations=None, maxr=90, nsample=9, crop_mesh=None):
        b, c, h, w = im.shape
        grid_3d = self.depth_to_3d_grid(depth)

        if crop_mesh is not None:
            top, bottom, left, right = crop_mesh  # pixels from border to be cropped
            if top > 0:
                grid_3d[:,:top,:,1] = grid_3d[:,top:top+1,:,1].repeat(1,top,1)
                grid_3d[:,:top,:,2] = grid_3d[:,top:top+1,:,2].repeat(1,top,1)
            if bottom > 0:
                grid_3d[:,-bottom:,:,1] = grid_3d[:,-bottom-1:-bottom,:,1].repeat(1,bottom,1)
                grid_3d[:,-bottom:,:,2] = grid_3d[:,-bottom-1:-bottom,:,2].repeat(1,bottom,1)
            if left > 0:
                grid_3d[:,:,:left,0] = grid_3d[:,:,left:left+1,0].repeat(1,1,left)
                grid_3d[:,:,:left,2] = grid_3d[:,:,left:left+1,2].repeat(1,1,left)
            if right > 0:
                grid_3d[:,:,-right:,0] = grid_3d[:,:,-right-1:-right,0].repeat(1,1,right)
                grid_3d[:,:,-right:,2] = grid_3d[:,:,-right-1:-right,2].repeat(1,1,right)

        grid_3d = grid_3d.reshape(b,-1,3)
        im_trans = []

        # inverse warp
        if v_before is not None:
            rot_mat, trans_xyz = get_transform_matrices(v_before)
            grid_3d = self.translate_pts(grid_3d, -trans_xyz)
            grid_3d = self.rotate_pts(grid_3d, rot_mat.transpose(2,1))

        if rotations is None:
            rotations = -torch.linspace(-math.pi/180*maxr, math.pi/180*maxr, nsample)
        for i, ri in enumerate(rotations):
            ri = torch.FloatTensor([0, ri, 0]).to(im.device).view(1,3)
            rot_mat_i, _ = get_transform_matrices(ri)
            grid_3d_i = self.rotate_pts(grid_3d, rot_mat_i.repeat(b,1,1))

            if v_after is not None:
                if len(v_after.shape) == 3:
                    v_after_i = v_after[i]
                else:
                    v_after_i = v_after
                rot_mat, trans_xyz = get_transform_matrices(v_after_i)
                grid_3d_i = self.rotate_pts(grid_3d_i, rot_mat)
                grid_3d_i = self.translate_pts(grid_3d_i, trans_xyz)

            faces = get_face_idx(b, h, w).to(im.device)
            rgb = im.permute(0,2,3,1).view(b,-1,3)
            warped_images = self.render_torch(grid_3d_i, faces, rgb).clamp(min=-1., max=1.)
            im_trans += [warped_images]
        return torch.stack(im_trans, 1)  # b x t x c x h x w
