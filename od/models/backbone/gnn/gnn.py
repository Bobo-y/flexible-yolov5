# -*- coding: utf-8 -*-
from unicodedata import name
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath

from .gcn_lib import Grapher, act_layer



class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x#.reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, num_k=9, conv='mr', bias=True, epsilon=0.2, stochastic=True, act='gelu', norm = 'batch', emb_dims=1024, drop_path=0.0, blocks=[2, 2, 6, 2], channels=[48, 96, 240, 384], img_size=[640, 640]):
        super(DeepGCN, self).__init__()
        num_k = num_k
        act = act
        norm = norm
        bias = bias
        epsilon = epsilon
        stochastic = stochastic
        conv = conv
        emb_dims = emb_dims
        drop_path = drop_path
        self.blocks = blocks
        self.n_blocks = sum(self.blocks)
        channels = channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(num_k, num_k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        h, w = img_size[0], img_size[1]
        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], h//4, w//4))

        HW = h // 4 * w // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.model_init()
        self.out_shape = [channels[-3],
                          channels[-2],
                          channels[-1]]
        print(self.out_shape)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        c3 = None
        c4 = None
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i == sum(self.blocks[:2]):
                c3  = x
            if i == sum(self.blocks[:3]) + 1:
                c4 = x
        return c3, c4, x


def tiny_gnn(pretrained=False, **kwargs):
    
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,6,2] # number of basic blocks in the backbone
    # channels = [48, 96, 240, 384] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings

    model = DeepGCN(**kwargs)
    return model


def small_gnn(pretrained=False, **kwargs):

    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,6,2] # number of basic blocks in the backbone
    # channels = [80, 160, 400, 640] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings

    model = DeepGCN(**kwargs)
    return model


def medium_gnn(pretrained=False, **kwargs):

    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,16,2] # number of basic blocks in the backbone
    # channels = [96, 182, 384, 768] # number of channels of deep features
    # emb_dims = 1024 # Dimension of embeddings


    model = DeepGCN(**kwargs)
    return model


def big_gnn(pretrained=False, **kwargs):
    # default params
    # num_k = 9 # neighbor num (default:9)
    # conv = 'mr' # graph conv layer {edge, mr}
    # act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
    # norm = 'batch' # batch or instance normalization {batch, instance}
    # bias = True # bias of conv layer True or False
    # dropout = 0.0 # dropout rate
    # use_dilation = True # use dilated knn or not
    # epsilon = 0.2 # stochastic epsilon for gcn
    # stochastic = False # stochastic for gcn, True or False
    # drop_path = 0.
    # blocks = [2,2,18,2] # number of basic blocks in the backbone
    # channels = [128, 256, 512, 1024] # number of channels of deep features

    model = DeepGCN(**kwargs)
    return model


def gnn(pretrained=False, **kwargs):
    version = kwargs.pop('version')
    if version == 'tiny':
        return tiny_gnn(pretrained, **kwargs)
    if version == 'small':
        return small_gnn(pretrained, **kwargs)
    if version == 'medium':
        return medium_gnn(pretrained, **kwargs)
    if version == 'big':
        return big_gnn(pretrained, **kwargs)


if __name__ == '__main__':
    model = gnn(pretrained=False, version ='small')
    model.eval()
    img = torch.rand(size=(1,3,640,640))
    c3, c4, c5 = model(img)
    print(c3.shape,c4.shape, c5.shape)
