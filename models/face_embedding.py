import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist
import sys

def group_norm(*args, **kwargs):
    return nn.GroupNorm(32, *args, **kwargs)

def build_norm(norm_type, dimension=2):
    if norm_type == 'batch':
        norm = nn.BatchNorm2d if dimension==2 else nn.BatchNorm1d
    elif norm_type == 'group':
        norm = group_norm
    else:
        raise ValueError('Unkown norm_type: {}'.format(norm_type))

    return norm


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                        nn.Linear(channel, channel // reduction),
                        nn.PReLU(),
                        nn.Linear(channel // reduction, channel),
                        nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):

        def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type='batch', use_se=True):
            super(IRBlock, self).__init__()

            norm_layer = build_norm(norm_type, dimension=2)

            self.bn0 = norm_layer(inplanes)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = norm_layer(planes)
            self.prelu = nn.PReLU(num_parameters=planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
            self.use_se = use_se
            if self.use_se:
                self.se = SEBlock(planes)

        def forward(self, x):
            residual = x
            out = self.bn0(x)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.prelu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if self.use_se:
                out = self.se(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out


class LResNet(nn.Module):
    def __init__(self, image_size=(112,112), nch=1, ndim=512, nlayer=50, width=None,
            norm_type='batch', dropout_rate=0.4, use_se=False):
        '''
                nch               : input channel (1 for gray scale, 3 for RGB)
                ndim              : embedding dimension
                nlayer            : total number of layers
        '''
        super(LResNet,self).__init__()

        if nlayer == 18:
            layers = [2,2,2,2]
        elif nlayer == 34:
            layers = [3,4,6,3]
        elif nlayer == 50:
            layers = [3,4,14,3]
        elif nlayer == 100:
            layers = [3,13,30,3]
        elif nlayer == 152:
            layers = [3,8,36,3]
        else:
            raise ValueError('Invalide nlayer: {}'.format(nlayer))

        if width == 'half':
            blocksize = [32, 64, 128, 256]
        elif width == 'one-half':
            blocksize = [96, 192, 384, 768]
        elif width == 'double':
            blocksize = [128, 256, 512, 1024]
        elif width =='single' or width is None:
            blocksize = [64, 128, 256, 512]
        else:
            raise ValueError('Invalid width: {}'.format(width))

        self.norm_layer = build_norm(norm_type, dimension=2)

        self.norm_type = norm_type
        self.use_se = use_se

        # first convolution block
        self.conv1 = nn.Conv2d(nch, blocksize[0], kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = self.norm_layer(blocksize[0])
        self.act_layer = nn.PReLU(num_parameters=blocksize[0])
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.inplanes = blocksize[0]

        # major blocks
        self.layer1 = self._make_layer(blocksize[0], layers[0], stride=1, norm_type=norm_type, use_se=use_se)

        self.layer2 = self._make_layer(blocksize[1], layers[1], stride=2, norm_type=norm_type, use_se=use_se)

        self.layer3 = self._make_layer(blocksize[2], layers[2], stride=2, norm_type=norm_type, use_se=use_se)

        self.layer4 = self._make_layer(blocksize[3], layers[3], stride=2, norm_type=norm_type, use_se=use_se)


        self.bn4 = self.norm_layer(blocksize[3])
        self.dropout = nn.Dropout(p=dropout_rate)

        h, w = self._get_feature_map_size(image_size, 4)
        self.fc5 = nn.Linear(blocksize[3] * h * w, ndim)
        self.bn5 = nn.BatchNorm1d(ndim)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _get_feature_map_size(self, image_size, downsample=4):
        h, w = image_size
        for i in range(4):
                h, w = math.ceil(0.5*h), math.ceil(0.5*w)
        h, w = int(h), int(w)
        return h, w

    def _make_layer(self, outplanes, nlayer, stride=1, norm_type='batch', use_se=True):

        norm_layer = build_norm(norm_type, dimension=2)
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
                downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False),
                        norm_layer(outplanes),
                )

        layers = []
        layers.append(IRBlock(self.inplanes, outplanes, stride, downsample, norm_type=norm_type, use_se=use_se))
        self.inplanes = outplanes
        for _ in range(1, nlayer):
            layers.append(IRBlock(self.inplanes, outplanes, norm_type=norm_type, use_se=use_se))
        return nn.Sequential(*layers)


    def forward(self, x):

        if x.size(2) == 256:
            x = F.avg_pool2d(x, 2, stride=2)
        else:
            assert x.size(2) == 128

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_layer(x)
        # x = self.maxpool(x)

        f1 = x = self.layer1(x)
        f2 = x = self.layer2(x)

        f3 = x = self.layer3(x)
        f4 = x = self.layer4(x)

        # x = F.avg_pool2d(x, tuple(x.shape[2:4]))

        x = self.bn4(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc5(x)
        x = self.bn5(x)

        x = F.normalize(x,p=2,dim=1)

        return x


def build_model(**args):
    return LResNet(**args)
