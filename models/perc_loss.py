import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EPS = 1e-7
class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False, loss_type='l2', n_scale=1, slice_indices=[2]):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)
        self.loss_type = loss_type
        self.n_scale = n_scale
        self.slice_indices = slice_indices[:]

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.classifier = nn.Conv2d(256, 1, 1, 1, bias=False)

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def compute_loss(self, f1, f2):
        if self.loss_type == 'l1':
            loss = (f1-f2).abs()
        elif self.loss_type == 'cosine':
            f1, f2 = F.normalize(f1,p=2,dim=1), F.normalize(f2,p=2,dim=1)
            loss = (f1-f2)**2
            loss = loss.sum(1, keepdim=True)
        else:
            loss = (f1-f2)**2
        return loss

    def resize(self, tensor, size):
        h, w = size
        h0, w0 = tensor.shape[2:]
        if h0==h and w0==w:
            return tensor
        assert h0%h==0 and w0%w==0
        sh, sw = h0//h, w0//w
        out = nn.functional.avg_pool2d(tensor, kernel_size=(sh,sw), stride=(sh,sw))
        return out

    def __call__(self, im1, im2, mask=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input
        losses = []
        input = im
        for i in range(self.n_scale):
            if i == 0:
                input = input
            else:
                input = F.avg_pool2d(input, 2, stride=2, padding=0)
            ## compute features
            feats = []
            f = self.slice1(input)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice2(f)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice3(f)
            feats += [torch.chunk(f, 2, dim=0)]
            f = self.slice4(f)
            feats += [torch.chunk(f, 2, dim=0)]

            for idx in self.slice_indices:  # use relu3_3 features only
                f1,f2 = feats[idx]
                loss = self.compute_loss(f1,f2)
                if mask is not None:
                    b, c, h, w = loss.shape
                    _, _, hm, wm = mask.shape
                    sh, sw = hm//h, wm//w
                    mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                    loss = (loss * mask0).sum() / mask0.sum()
                else:
                    loss = loss.mean()
                losses += [loss]
        return sum(losses) / len(losses)
