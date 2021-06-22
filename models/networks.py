import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



class ScaleShift(nn.Module):
    def __init__(self, num_features, scale=True, shift=True, init_scale=1.0):
        super().__init__()
        self.scale = scale
        self.shift = shift
        if self.scale:
            self.weight = nn.Parameter(init_scale*torch.ones(1,num_features).cuda())
        if self.shift:
            self.bias = nn.Parameter(torch.zeros(1,num_features).cuda())

    def forward(self, x):
        if self.scale:
            x = self.weight * x
        if self.shift:
            x = self.bias + x
        return x


class MLP(nn.Module):
    def __init__(self, cin, cout, n_layer=4, dim=512, normalize=False, activation=None):
        super().__init__()

        mlp = []
        for i in range(n_layer-1):
            dim_in = cin if i==0 else dim
            mlp.append(nn.Linear(dim_in, dim))
            if normalize:
                mlp.append(nn.GroupNorm(dim//4, dim))
            mlp.append(nn.LeakyReLU(0.2, inplace=True))
        mlp.append(nn.Linear(dim, cout))
        if activation is not None:
            mlp.append(activation)
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        out = self.mlp(x)
        return out


class StyleDecomposeNet(nn.Module):
    def __init__(self, style_dim, light_dim=4, view_dim=6, n_layer=4):
        super().__init__()
        self.model_style = MLP(style_dim, style_dim, n_layer=n_layer, normalize=True, activation=None)
        self.model_light = MLP(style_dim, light_dim, n_layer=n_layer, activation=nn.Tanh())
        self.model_view = MLP(style_dim, view_dim, n_layer=n_layer, activation=nn.Tanh())

    def forward(self, style):
        neutral_style = self.model_style(style)
        light = self.model_light(style)
        view = self.model_view(style)
        return neutral_style, light, view

class StyleComposeNet(nn.Module):
    def __init__(self, style_dim, light_dim=4, view_dim=6, n_layer=4):
        super().__init__()
        self.embed_style = MLP(style_dim, style_dim, n_layer=4, normalize=True, activation=ScaleShift(style_dim))
        self.embed_light = MLP(light_dim, style_dim, n_layer=4, normalize=True, activation=ScaleShift(style_dim))
        self.embed_view = MLP(view_dim, style_dim, n_layer=4, normalize=True, activation=ScaleShift(style_dim))
        self.model = MLP(style_dim, style_dim, n_layer=4, normalize=True, activation=None)

    def forward(self, neutral_style, light, view):
        out_shape = self.embed_style(neutral_style)
        out_light = self.embed_light(light)
        out_view = self.embed_view(view)
        out = out_shape + out_light + out_view
        style = self.model(out)
        return style


class DepthNet(nn.Module):
    def __init__(self, in_dim, out_dim, inject_dim=16, z_dim=128, nf=16, n_mlp=0, activation=nn.Tanh, bias=False):
        super().__init__()
        # MLP
        self.n_mlp = n_mlp
        if n_mlp > 0:
            mlp = []
            for i in range(n_mlp):
                mlp.append(nn.Linear(in_dim, z_dim))
                mlp.append(nn.LeakyReLU(0.2, inplace=True))
                in_dim = z_dim
            self.mlp = nn.Sequential(*mlp)
        # upsampling
        network = []
        ndim = nf*32
        network += [
            nn.ConvTranspose2d(z_dim, ndim, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, ndim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)]
        for i in range(5):
            ndim = ndim // 2
            network += [
                nn.ConvTranspose2d(ndim*2, ndim, kernel_size=4, stride=2, padding=1, bias=False),  # 1x1 -> 4x4
                nn.GroupNorm(ndim//4, ndim),
                nn.ReLU(inplace=True),
                nn.Conv2d(ndim, ndim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(ndim//4, ndim),
                nn.ReLU(inplace=True)]
        network += [nn.Upsample(scale_factor=2, mode='nearest')]
        self.network = nn.Sequential(*network)

        self.inject = nn.Conv2d(inject_dim, ndim, kernel_size=1, stride=1, padding=0, bias=False)

        head = [
            nn.Conv2d(ndim, ndim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(ndim//4, ndim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, ndim, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(ndim//4, ndim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, out_dim, kernel_size=5, stride=1, padding=2, bias=True)
        ]
        if activation is not None:
            head += [activation()]
        self.head = nn.Sequential(*head)

    def forward(self, input, input2):
        if self.n_mlp > 0:
            out = self.mlp(input)
        else:
            out = input
        out = out.view(out.size(0), out.size(1), 1 ,1)
        out = self.network(out) + self.inject(input2)
        return self.head(out)


class TransformationNet(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=128, nf=8, n_mlp=0, activation=nn.Tanh):
        super().__init__()
        # MLP
        self.n_mlp = n_mlp
        if n_mlp > 0:
            mlp = []
            for i in range(n_mlp):
                mlp.append(nn.Linear(in_dim, z_dim))
                mlp.append(nn.LeakyReLU(0.2, inplace=True))
                in_dim = z_dim
            self.mlp = nn.Sequential(*mlp)
        # upsampling
        network = [
            nn.ConvTranspose2d(z_dim, nf*32, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True)]
        indim, outdim = nf*32, nf*16
        for i in range(5):
            network += [
                nn.ConvTranspose2d(indim, outdim, kernel_size=4, stride=2, padding=1, bias=False),  # 1x1 -> 4x4
                nn.GroupNorm(outdim//4, outdim),
                nn.ReLU(inplace=True)]
            indim, outdim = outdim, outdim//2
        network += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(nf//4, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_dim, kernel_size=5, stride=1, padding=2, bias=False)]
        # if bias:
        #     network += [SpatialBias(128, 1, 'centered')]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        if self.n_mlp > 0:
            out = self.mlp(input)
        else:
            out = input
        out = out.view(out.size(0), out.size(1), 1 ,1)
        out = self.network(out)
        return out



class Blur(nn.Module):
    def __init__(self, size, sigma=2.0, pad_mode='reflect'):
        super().__init__()
        self.kernel = self.gaussian_kernel(size=size, sigma=sigma).cuda()
        self.kernel_size = 2*size + 1
        self.pad_mode = pad_mode

    def gaussian_kernel(self, size, sigma, dim=2, channels=3):
        # The gaussian kernel is the product of the gaussian function of each dimension.
        # kernel_size should be an odd number.

        kernel_size = 2*size + 1

        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return kernel

        return x

    def forward(self, x):
        padding = int((self.kernel_size - 1) / 2)
        kernel = self.kernel.repeat(x.size(1), 1, 1,1)
        x = F.pad(x, (padding, padding, padding, padding), mode=self.pad_mode)
        x = F.conv2d(x, kernel, groups=x.size(1))
        return x


class Laplacian(nn.Module):
    def __init__(self, size=0, sigma=1.0):
        super().__init__()
        if size>0:
            self.blur_kernel = Blur(size, sigma=sigma)
        else:
            self.blur_kernel = None
        self.kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).cuda()
        self.kernel = self.kernel.view(1, 1, 3, 3) / 4

    def forward(self, x):
        if self.blur_kernel:
            x = self.blur_kernel(x)
        kernel = self.kernel.repeat(x.size(1), 1, 1,1)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x, kernel, groups=x.shape[1])


class ContourGradientClip(Function):
    laplacian = Laplacian(size=0,sigma=1.0)
    mask = None

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        eps = 1e-5
        laplacian = ContourGradientClip.laplacian
        mask = torch.abs(grad_output[:,None]) > 0
        contour = torch.abs(laplacian(mask.float())) < eps
        contour = contour.squeeze(1)
        # mask = mask & contour
        # ContourGradientClip.mask = contour.float()
        # ContourGradientClip.grad = contour.float() * grad_output
        return  contour.float() * grad_output
