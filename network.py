from turtle import pos
import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from argparse import Namespace


class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling. T
    """

    def __init__(self, args: Namespace, cluster: bool = False):
        super().__init__()

        self.backbone = get_backbone(args)
        if not cluster:
            if args.layer == "avg":
                self.aggregation = nn.Sequential(L2Norm(),
                                                 torch.nn.AdaptiveAvgPool2d(1),
                                                 Flatten())
            elif args.layer == "net":
                self.aggregation = NetVLAD(args.clusters)
            elif args.layer == "gem":
                self.aggregation = nn.Sequential(GeM(args), L2Norm(), Flatten())
        else:
            # Redundant, however, may be useful if we were to change the aggregation method for obtaining centroids, if not found just refactor
            self.aggregation = nn.Sequential(
                # L2Norm(),
                torch.nn.AdaptiveAvgPool2d(1),
                Flatten())

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_backbone(args):
    backbone = torchvision.models.resnet18(pretrained=True, progress=False)
    for name, child in backbone.named_children():
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    logging.debug(
        "Train only conv4 of the ResNet-18 (remove conv5), freeze the previous ones"
    )
    layers = list(backbone.children())[:-3]
    backbone = torch.nn.Sequential(*layers)

    # features_dim is used in datasets_ws.compute_cache to build a cache of proper size. Not modifying it to fit current output leads to error
    if args.layer == "avg" or args.layer == "gem" or args.layer == "solar":
        args.features_dim = 256  # Number of channels in conv4
    elif args.layer == "net":
        args.features_dim = (
                256 * args.num_clusters
        )  # NetVLAD should output a KxDx1 (?), with K = num_clusters and D = features of local descriptor xi
    return backbone


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class NetVLAD(nn.Module):
    def __init__(self, num_clusters):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = 256  # should be the channels given by conv4

        # 1x1 convolution
        self.conv = nn.Conv2d(self.dim, num_clusters, kernel_size=(1, 1))
        # cluster centers are also learnable parameters (initialized randomly)
        self.centroids = nn.Parameter(torch.rand(num_clusters, self.dim))

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray):
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        N, C = x.shape[:2]

        x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros(
            [N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device
        )
        for C in range(
                self.num_clusters
        ):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[
                                                                    C: C + 1, :
                                                                    ].expand(x_flatten.size(-1), -1, -1).permute(1, 2,
                                                                                                                 0).unsqueeze(
                0)
            residual *= soft_assign[:, C: C + 1, :].unsqueeze(2)
            vlad[:, C: C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class GeM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pk = torch.nn.Parameter(torch.zeros(1) + args.pk)
        self.minval = args.minval

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.minval).pow(self.pk), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.pk)


# The following lines of code were obtained from https://github.com/tonyngjichun/SOLAR


class SOA(nn.Module):

    def __init__(self, in_ch, k):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        self.f = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]:
            conv.apply(weights_init)

        self.v.apply(constant_init)

    def forward(self, x, vis_mode=False):
        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W

        if vis_mode:
            # for visualisation only
            attn = self.softmax((self.mid_ch ** -.75) * z)
        else:
            attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W

        z = self.v(z)
        z = z + x

        return z


def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass


# class LocalWhitening(nn.Module):
# Doesn't work because matrices might be singular
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         x_tmp = x[:, :, 0, 0]  # Making the input a 2D matrix
#         x_tmp = F.normalize(x_tmp)  # Normalizing the input
#         cov = torch.mm(x_tmp, x_tmp.t())  # Computing the covariance matrix
#         evalue, evec = torch.eig(cov, eigenvectors=True)  # Computing the eigenvalues and eigenvectors
#         lambda_rsqrt = torch.rsqrt(evalue[:, 0])
#         lambda_rsqrt = torch.diag(lambda_rsqrt)
#         res = evec @ lambda_rsqrt @ evec.t()  # ZCA Whitening
#         y = res @ x_tmp
#
#         return y.unsqueeze(-1).unsqueeze(-1)

class SVDWhitening(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_tmp = F.normalize(x)  # Normalizing the input
        # cov = torch.mm(x_tmp, x_tmp.t())  # Computing the covariance matrix
        U, _, V = torch.svd(x_tmp)
        y = U @ V.t()
        return y

class SVDWhiteningCov(nn.Module):


    def __init__(self):
            super().__init__()

    def forward(self, x):
        x_tmp = F.normalize(x)  # Normalizing the input
        cov = torch.mm(x_tmp, x_tmp.t())  # Computing the covariance matrix
        U, s, V = torch.svd(cov)
        lambda_rsqrt = torch.diag(torch.rsqrt(s))
        y = U @ lambda_rsqrt @ V.t()
        return y @ x_tmp

class GeoLocalizationNetSOA(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.backbone = get_backbone(args)

        self.resnet_fixed = self.backbone[0:6]
        self.resnet_2 = self.backbone[6]

        self.soa1 = SOA(in_ch=args.features_dim // 2, k=4)
        self.soa2 = SOA(in_ch=args.features_dim, k=2)

        self.gem = GeM(args)

        if args.solar_whiten:
            self.whiten = nn.Linear(args.features_dim, args.features_dim, bias=True)
        else:
            self.whiten = None

        # self.whiten = SVDWhiteningCov()

        self.aggregation = nn.Sequential(L2Norm(), Flatten())

    def forward(self, x):
        x = self.resnet_fixed(x) # First part of the Resnet-18 (Fixed weights)
        # According to the solar paper this is the arrangement that worked best
        x = self.soa1(x)
        x = self.resnet_2(x)
        x = self.soa2(x)

        #Normalization and aggregation step
        x = self.gem(x)
        if self.whiten is not None:
            x = self.whiten(x[:, :, 0, 0])
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.aggregation(x)

        return x


def fos_triplet_loss(query,positives,negatives, margin=0.1):
    """UNUSED"""
    # x is D x N
    #dim = query.size(0) # D
    nq = query.size(0) # number of tuples
    #S = 1+1+negatives.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = query #.permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xp = positives #.permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = negatives

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))/nq
  
def sos_loss(query,positives, negatives):
    # x is D x N
    #dim = query.size(1) # D
    nq = query.size(0) # number of tuples
    #S = 1+1+negatives.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = query
    xp = positives
    xn = negatives

    dist_an = torch.sum(torch.pow(xa - xn, 2), dim=0)
    dist_pn = torch.sum(torch.pow(xp - xn, 2), dim=0)

    return torch.sum(torch.pow(dist_an - dist_pn, 2))

