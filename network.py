import torch
import logging
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class GeoLocalizationNet(nn.Module):
    """The model is composed of a backbone and an aggregation layer.
    The backbone is a (cropped) ResNet-18, and the aggregation is a L2
    normalization followed by max pooling. T
    """

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        if args.layer == "avg":
            self.aggregation = nn.Sequential(
                L2Norm(), torch.nn.AdaptiveAvgPool2d(1), Flatten()
            )
        elif args.layer == "net":
            self.aggregation = NetVLAD(args.num_clusters)
        elif args.layer == "gem":
            self.aggregation = nn.Sequential(GeM(args), L2Norm())

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
    if args.layer == "avg" or args.layer == "gem":
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
                C : C + 1, :
            ].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C : C + 1, :].unsqueeze(2)
            vlad[:, C : C + 1, :] = residual.sum(dim=-1)

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
