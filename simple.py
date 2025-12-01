import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from kornia.utils import create_meshgrid
from einops import rearrange
import math
from mamba_ssm import Mamba
from functools import partial
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None
from pytorch_lightning.profiler import SimpleProfiler, PassThroughProfiler
from loguru import logger

torch.backends.cudnn.deterministic = True
INF = 1e9


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

class GLU_3(nn.Module):
    def __init__(self, dim, mid_dim):
        super(GLU_3, self).__init__()
        self.W = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.V = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(mid_dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(True)
        self.act = nn.GELU()

    def forward(self, feat):
        feat_act = self.act(self.W(feat))
        feat_linear = self.V(feat)
        feat = feat_act * feat_linear
        feat = self.W2(feat)
        return feat
        # return self.V(feat)

class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, desc, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            desc = desc.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return desc + hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def scan_jego(desc0, desc1, step_size):
    desc_2w, desc_2h = torch.cat([desc0, desc1], 3), torch.cat([desc0, desc1], 2)
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H*W))

    xs[:, 0] = desc_2w[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)  # [h/2, 2w/2]
    xs[:, 1] = desc_2h.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)  # [w/2, 2w/2]
    xs[:, 2] = desc_2w[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [h/2, 2w/2]
    xs[:, 3] = desc_2h.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])  # [w/2, 2w/2]

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w

def merge_jego(ys, ori_h: int, ori_w: int, step_size=2):
    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))

    y_2w[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)
    y_2w[:, :, ::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, H, 2*W)
    y_2h[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, W, 2*H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h


    B, K, C, L = ys.shape
    H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)

    new_h = H * step_size
    new_w = W * step_size

    y_2w = torch.zeros((B, C, new_h, 2*new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, new_h, 2*new_w))
    y_2h = torch.zeros((B, C, 2*new_h, new_w), device=ys.device, dtype=ys.dtype)  # ys.new_empty((B, C, 2*new_h, new_w))

    y_2h[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, 2*H, W)
    y_2w[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, 2*W, H).transpose(dim0=2, dim1=3)
    y_2h[:, :, ::step_size, 1::step_size] = ys[:, 2].flip([2]).reshape(B, C, 2*H, W)
    y_2w[:, :, 1::step_size, ::step_size] = ys[:, 3].flip([2]).reshape(B, C, 2*W, H).transpose(dim0=2, dim1=3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()
    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w+desc0_2h, desc1_2w+desc1_2h

class JointMamba(nn.Module):
    def __init__(self, feature_dim: int, depth,
                 ssm_cfg=None,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 profiler=None):
        super().__init__()
        self.profiler = profiler or PassThroughProfiler()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_layers = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(create_block(
                    feature_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                ))
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.aggregator = GLU_3(feature_dim, feature_dim)

    def forward(self, data):
        desc0, desc1 = data['feat_8_0'], data['feat_8_1']
        desc0, desc1 = desc0.view(data['bs'], -1, data['h_8'], data['w_8']), desc1.view(data['bs'], -1, data['h_8'], data['w_8'])
        x, ori_h, ori_w = scan_jego(desc0, desc1, 2)
        for i in range(len(self.layers) // 4):
            y0 = self.layers[i * 4](x[:, 0])
            y1 = self.layers[i * 4 + 1](x[:, 1])
            y2 = self.layers[i * 4 + 2](x[:, 2])
            y3 = self.layers[i * 4 + 3](x[:, 3])
            y = torch.stack([y0, y1, y2, y3], 1).transpose(2, 3)
        desc0, desc1 = merge_jego(y, ori_h, ori_w, 2)
        desc = self.aggregator(torch.cat([desc0, desc1], 0))
        desc0, desc1 = torch.chunk(desc, 2, dim=0)
        desc0, desc1 = desc0.flatten(2, 3), desc1.flatten(2, 3)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v

def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

def generate_random_mask(n, num_true):
    # 创建全 False 的掩码
    mask = torch.zeros(n, dtype=torch.bool)

    # 随机选择 num_true 个位置并设置为 True
    indices = torch.randperm(n)[:num_true]
    mask[indices] = True

    return mask

class CoarseMatching(nn.Module):
    def __init__(self, config, profiler):
        super().__init__()
        self.config = config
        # general config
        d_model = 256
        self.thr = config['thr']
        self.use_sm = config['use_sm']
        self.inference = config['inference']
        self.border_rm = config['border_rm']

        self.final_proj = nn.Linear(d_model, d_model, bias=True)

        self.temperature = config['dsmax_temperature']
        self.profiler = profiler

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_c0, feat_c1])

        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                  feat_c1) / self.temperature
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                -INF)
        if self.inference:
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_inference(sim_matrix, data))
        else:
            conf_matrix_0_to_1 = F.softmax(sim_matrix, 2)
            conf_matrix_1_to_0 = F.softmax(sim_matrix, 1)
            data.update({'conf_matrix_0_to_1': conf_matrix_0_to_1,
                         'conf_matrix_1_to_0': conf_matrix_1_to_0
                         })
            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match_training(conf_matrix_0_to_1, conf_matrix_1_to_0, data))

    @torch.no_grad()
    def get_coarse_match_training(self, conf_matrix_0_to_1, conf_matrix_1_to_0, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix_0_to_1.device

        # confidence thresholding
        # {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(
            (conf_matrix_0_to_1 > self.thr) * (conf_matrix_0_to_1 == conf_matrix_0_to_1.max(dim=2, keepdim=True)[0]),
            (conf_matrix_1_to_0 > self.thr) * (conf_matrix_1_to_0 == conf_matrix_1_to_0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = torch.maximum(conf_matrix_0_to_1[b_ids, i_ids, j_ids], conf_matrix_1_to_0[b_ids, i_ids, j_ids])

        # random sampling of training samples for fine-level XoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # the sampling is performed across all pairs in a batch without manually balancing
            # samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.config['train_coarse_percent'])
            num_matches_pred = len(b_ids)
            train_pad_num_gt_min = self.config['train_pad_num_gt_min']
            assert train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - train_pad_num_gt_min,),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                len(data['spv_b_ids']),
                (max(num_matches_train - num_matches_pred,
                     train_pad_num_gt_min),),
                device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mkpts0_c_train': mkpts0_c,
            'mkpts1_c_train': mkpts1_c,
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    @torch.no_grad()
    def get_coarse_match_inference(self, sim_matrix, data):
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 2) if self.use_sm else sim_matrix

        # confidence thresholding and nearest neighbour for 0 to 1
        mask = (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=2, keepdim=True)[0])

        # unlike training, reuse the same conf martix to decrease the vram consumption
        # softmax for 0 to 1
        conf_matrix_ = F.softmax(sim_matrix, 1) if self.use_sm else sim_matrix

        # update mask {(nearest neighbour for 0 to 1) U (nearest neighbour for 1 to 0)}
        mask = torch.logical_or(mask,
                                (conf_matrix_ > self.thr) * (conf_matrix_ == conf_matrix_.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # find all valid coarse matches
        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)

        mconf = sim_matrix[b_ids, i_ids, j_ids]

        # these matches are selected patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # these matches are the current coarse level predictions
        coarse_matches.update({
            'mconf': mconf,
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
        })

        return coarse_matches

class FineSubMatching(nn.Module):
    """Fine-level and Sub-pixel matching"""

    def __init__(self, config, profiler):
        super().__init__()
        self.temperature = config['fine']['dsmax_temperature']
        self.W_f = config['fine_window_size']
        self.inference = config['fine']['inference']
        dim_f = 64
        self.fine_thr = config['fine']['thr']
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(2 * dim_f, 4, bias=False))
        self.fine_spv_max = 500  # saving memory
        self.profiler = profiler

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        M, WW, C = feat_f0_unfold.shape
        W_f = self.W_f

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            logger.warning('No matches found in coarse-level.')
            if self.inference:
                data.update({
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                })
            else:
                data.update({
                    'mkpts0_f': data['mkpts0_c'],
                    'mkpts1_f': data['mkpts1_c'],
                    'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
                    'mkpts0_f_train': data['mkpts0_c_train'],
                    'mkpts1_f_train': data['mkpts1_c_train'],
                    'conf_matrix_fine': torch.zeros(1, W_f * W_f, W_f * W_f, device=feat_f0_unfold.device),
                    'b_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'i_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'j_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                })
            return

        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        # normalize
        feat_f0, feat_f1 = map(lambda feat: feat / feat.shape[-1] ** .5,
                               [feat_f0, feat_f1])
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0,
                                  feat_f1) / self.temperature

        conf_matrix_fine = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # predict fine-level and sub-pixel matches from conf_matrix
        data.update(**self.get_fine_sub_match(conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data))

    def get_fine_sub_match(self, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data):
        with torch.no_grad():
            W_f = self.W_f

            # 1. confidence thresholding
            mask = conf_matrix_fine > self.fine_thr

            if mask.sum() == 0:
                mask[0, 0, 0] = 1
                conf_matrix_fine[0, 0, 0] = 1

            # match only the highest confidence
            mask = mask \
                   * (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

            # 3. find all valid fine matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

            # 4. update with matches in original image resolution

            # indices from coarse matches
            b_ids_c, i_ids_c, j_ids_c = data['b_ids'], data['i_ids'], data['j_ids']

            # scale (coarse level / fine-level)
            scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

            # coarse level matches scaled to fine-level (1/2)
            mkpts0_c_scaled_to_f = torch.stack(
                [i_ids_c % data['hw0_c'][1], torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            mkpts1_c_scaled_to_f = torch.stack(
                [j_ids_c % data['hw1_c'][1], torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc')],
                dim=1) * scale_f_c

            # updated b_ids after second thresholding
            updated_b_ids = b_ids_c[b_ids]

            # scales (image res / fine level)
            scale = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

            # fine-level discrete matches on window coordiantes
            mkpts0_f_window = torch.stack(
                [i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')],
                dim=1)

            mkpts1_f_window = torch.stack(
                [j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')],
                dim=1)

        # sub-pixel refinement
        sub_ref = self.subpixel_mlp(torch.cat([feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]], dim=-1))
        sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
        sub_ref0, sub_ref1 = sub_ref0.squeeze(1), sub_ref1.squeeze(1)
        sub_ref0 = torch.tanh(sub_ref0) * 0.5
        sub_ref1 = torch.tanh(sub_ref1) * 0.5

        pad = 0 if W_f % 2 == 0 else W_f // 2
        # final sub-pixel matches by (coarse-level + fine-level windowed + sub-pixel refinement)
        mkpts0_f1 = (mkpts0_f_window + mkpts0_c_scaled_to_f[b_ids] - pad) * scale0  # + sub_ref0
        mkpts1_f1 = (mkpts1_f_window + mkpts1_c_scaled_to_f[b_ids] - pad) * scale1  # + sub_ref1
        mkpts0_f_train = mkpts0_f1 + sub_ref0 * scale0  # + sub_ref0
        mkpts1_f_train = mkpts1_f1 + sub_ref1 * scale1  # + sub_ref1
        mkpts0_f = mkpts0_f_train.clone().detach()
        mkpts1_f = mkpts1_f_train.clone().detach()

        # These matches is the current prediction (for visualization)
        sub_pixel_matches = {
            'm_bids': b_ids_c[b_ids[mconf != 0]],  # mconf == 0 => gt matches
            'mkpts0_f1': mkpts0_f1[mconf != 0],
            'mkpts1_f1': mkpts1_f1[mconf != 0],
            'mkpts0_f': mkpts0_f[mconf != 0],
            'mkpts1_f': mkpts1_f[mconf != 0],
            'mconf_f': mconf[mconf != 0]
        }

        # These matches are used for training
        if not self.inference:
            if self.fine_spv_max is None or self.fine_spv_max > len(data['b_ids']):
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'],
                    'i_ids_fine': data['i_ids'],
                    'j_ids_fine': data['j_ids'],
                    'conf_matrix_fine': conf_matrix_fine
                })
            else:
                train_mask = generate_random_mask(len(data['b_ids']), self.fine_spv_max)
                sub_pixel_matches.update({
                    'mkpts0_f_train': mkpts0_f_train,
                    'mkpts1_f_train': mkpts1_f_train,
                    'b_ids_fine': data['b_ids'][train_mask],
                    'i_ids_fine': data['i_ids'][train_mask],
                    'j_ids_fine': data['j_ids'][train_mask],
                    'conf_matrix_fine': conf_matrix_fine[train_mask]
                })

        return sub_pixel_matches

class KeypointEncoder_wo_score(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
        kpts: torch.Tensor,
        size: torch.Tensor) -> torch.Tensor:
    if not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts

class up_conv4(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(up_conv4, self).__init__()
        self.lin = nn.Conv2d(dim_in, dim_mid, kernel_size=3, stride=1, padding=1)
        self.inter = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transconv = nn.ConvTranspose2d(dim_in, dim_mid, kernel_size=2, stride=2)
        self.cbr = nn.Sequential(
            nn.Conv2d(dim_mid, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_inter = self.inter(self.lin(x))
        x_conv = self.transconv(x)
        x = self.cbr(x_inter+x_conv)
        return x

class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, dim1, dim2, factor=1):
        super(MLPMixerEncoderLayer, self).__init__()

        self.mlp1 = nn.Sequential(nn.Linear(dim1, dim1*factor),
                                  nn.GELU(),
                                  nn.Linear(dim1*factor, dim1))
        self.mlp2 = nn.Sequential(nn.Linear(dim2, dim2*factor),
                                  nn.LayerNorm(dim2*factor),
                                  nn.GELU(),
                                  nn.Linear(dim2*factor, dim2))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            x_mask (torch.Tensor): [N, L] (optional)
        """
        x = x + self.mlp1(x)
        x = x.transpose(1, 2)
        x = x + self.mlp2(x)
        return x.transpose(1, 2)

class JamMa(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        self.kenc = KeypointEncoder_wo_score(self.d_model_c, [32, 64, 128, self.d_model_c])
        self.joint_mamba = JointMamba(self.d_model_c, 4, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, profiler=self.profiler)
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler)

        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2*dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, stride=1, padding=1)
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, stride=1, padding=1)

        W = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([MLPMixerEncoderLayer(2*W**2, 64) for _ in range(4)])
        self.fine_matching = FineSubMatching(config, self.profiler)

    def coarse_match(self, data):
        desc0, desc1 = data['feat_8_0'].flatten(2, 3), data['feat_8_1'].flatten(2, 3)
        kpts0, kpts1 = data['grid_8'], data['grid_8']
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['imagec_0'].shape[-2:])
        kpts1 = normalize_keypoints(kpts1, data['imagec_1'].shape[-2:])

        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        desc0 = desc0 + self.kenc(kpts0)
        desc1 = desc1 + self.kenc(kpts1)
        data.update({
            'feat_8_0': desc0,
            'feat_8_1': desc1,
        })

        with self.profiler.profile("coarse interaction"):
            self.joint_mamba(data)

        # 3. match coarse-level
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        with self.profiler.profile("coarse matching"):
            self.coarse_matching(data['feat_8_0'].transpose(1,2), data['feat_8_1'].transpose(1,2), data, mask_c0=mask_c0, mask_c1=mask_c1)

    def inter_fpn(self, feat_8, feat_4):
        d2 = self.up2(feat_8)  # 1/4
        d2 = self.act(self.conv7a(torch.cat([feat_4, d2], 1)))
        feat_4 = self.act(self.conv7b(d2))

        d1 = self.up3(feat_4)  # 1/2
        d1 = self.act(self.conv8a(d1))
        feat_2 = self.conv8b(d1)
        return feat_2

    def fine_preprocess(self, data, profiler):
        data['resolution1'] = 8
        stride = data['resolution1'] // self.config['resolution'][1]
        W = self.config['fine_window_size']
        feat_8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(2*data['bs'], data['c'], data['h_8'], -1)
        feat_4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)

        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            feat1 = torch.empty(0, W ** 2, self.d_model_f, device=feat_4.device)
            return feat0, feat1

        # feat_f = self.inter_fpn(feat_8, feat_4, feat_2)
        feat_f = self.inter_fpn(feat_8, feat_4)
        feat_f0, feat_f1 = torch.chunk(feat_f, 2, dim=0)
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # 1. unfold(crop) all local windows
        pad = 0 if W % 2 == 0 else W//2
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=pad)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W ** 2)  # [b, h_f/stride * w_f/stride, w*w, c]

        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]  # [n, ww, cf]

        feat_f = torch.cat([feat_f0_unfold, feat_f1_unfold], 1).transpose(1, 2)
        for layer in self.fine_enc:
            feat_f = layer(feat_f)
        feat_f0_unfold, feat_f1_unfold = feat_f[:, :, :W**2], feat_f[:, :, W**2:]
        return feat_f0_unfold, feat_f1_unfold

    def forward(self, data, mode='test'):
        self.mode = mode
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        self.coarse_match(data)

        with self.profiler.profile("fine matching"):
            # 4. fine-level matching module
            feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(data, self.profiler)

            # 5. match fine-level and sub-pixel refinement
            self.fine_matching(feat_f0_unfold.transpose(1, 2), feat_f1_unfold.transpose(1, 2), data)


