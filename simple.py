import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_8(self, x):
        feat = {}
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feat[4] = x
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        feat[8] = x
        return feat  # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_4(self, x):
        feat = {}
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        feat[4] = x
        return feat  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class CovNextV2_nano(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640])
        self.cnn.norm = None
        self.cnn.head = None
        self.cnn.downsample_layers[2] = None
        self.cnn.downsample_layers[3] = None
        self.cnn.stages[2] = None
        self.cnn.stages[3] = None

        state_dict = torch.hub.load_state_dict_from_url(
            'https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt',
            file_name='convnextv2_nano_pretrain.ckpt')
        self.cnn.load_state_dict(state_dict, strict=True)

        self.lin_4 = nn.Conv2d(80, 128, 1)
        self.lin_8 = nn.Conv2d(160, 256, 1)

    def forward(self, data):
        B, _, H, W = data['imagec_0'].shape
        x = torch.cat([data['imagec_0'], data['imagec_1']], 0)
        feature_pyramid = self.cnn.forward_features_8(x)
        feat_8_0, feat_8_1 = self.lin_8(feature_pyramid[8]).split(B)
        feat_4_0, feat_4_1 = self.lin_4(feature_pyramid[4]).split(B)

        scale = 8
        h_8, w_8 = H//scale, W//scale
        device = data['imagec_0'].device
        grid = [rearrange((create_meshgrid(h_8, w_8, False, device) * scale).squeeze(0), 'h w t->(h w) t')] * B  # kpt_xy
        grid_8 = torch.stack(grid, 0)

        data.update({
            'bs': B,
            'c': feat_8_0.shape[1],
            'h_8': h_8,
            'w_8': w_8,
            'hw_8': h_8 * w_8,
            'feat_8_0': feat_8_0,
            'feat_8_1': feat_8_1,
            'feat_4_0': feat_4_0,
            'feat_4_1': feat_4_1,
            'grid_8': grid_8,
        })