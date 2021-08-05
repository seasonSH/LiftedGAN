import os
import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
from imageio import imread


from models.lifted_gan import LiftedGAN
from utils import utils
from utils.calc_inception import load_patched_inception_v3



@torch.no_grad()
def extract_feature_from_generator(
    model, inception, batch_size, n_sample, truncation=1.0):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch
    if resid > 0: batch_sizes.append(resid)

    features = []
    for batch in tqdm(batch_sizes):
        code = torch.randn(batch, 512).cuda()
        styles = model.generator.style(code)
        styles = truncation * styles + (1-truncation) * model.w_mu
        canon_depth, canon_albedo, canon_light, view, neutral_style, trans_map, canon_im_raw = model.estimate(styles)
        recon_im = model.render(canon_depth, canon_albedo, canon_light, view, trans_map=trans_map)[0]
        img = recon_im.clamp(min=-1,max=1)

        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)
    return features

@torch.no_grad()
def extract_feature_from_images(
    paths, inception, batch_size, n_sample, truncation=1.0):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch 
    if resid > 0: batch_sizes.append(resid)

    features = []
    count=0
    for batch in tqdm(batch_sizes):
        paths_batch = paths[count:count+batch]
        images = [imread(p) for p in paths_batch]
        img = torch.tensor(np.stack(images,0)).permute(0,3,1,2) / 127.5 - 1.0
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))
        count += batch_size

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help="path to the model or image folder")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--inception', type=str, default='pretrained/inception_ffhq_cropped_256.pkl')

    args = parser.parse_args()

    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    if args.input.endswith('.pth'):
        model = LiftedGAN()
        model.load_model(args.input)
        features = extract_feature_from_generator(
            model, inception, batch_size=args.batch_size, n_sample=args.n_sample).numpy()
    else:
        paths = [os.path.join(args.input, f) for f in os.listdir(args.input)]
        assert len(paths) >= args.n_sample
        features = extract_feature_from_images(
            paths, inception, batch_size=args.batch_size, n_sample=args.n_sample).numpy()

    print(f'extracted {features.shape[0]} features')

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print('fid:', fid)
