import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from importlib import import_module
import numpy as np


def make_model(opt):
    """Create DeAltHDR model with dual encoder and mixed training support"""
    model_config = {
        'inp_channels': opt['n_colors'],
        'out_channels': opt['n_colors'],
        'dim': opt['dim'],
        'Enc_blocks': opt['Enc_blocks'],
        'Middle_blocks': opt['Middle_blocks'],
        'Dec_blocks': opt['Dec_blocks'],
        'num_refinement_blocks': opt.get('num_refinement_blocks', 1),
        'ffn_expansion_factor': opt.get('ffn_expansion_factor', 1),
        'bias': opt.get('bias', False),
        'LayerNorm_type': opt.get('LayerNorm_type', 'WithBias'),
        'num_heads_blks': opt.get('num_heads_blks', [1,2,4,8]),
        'encoder1_attn_type1': opt['encoder1_attn_type1'],
        'encoder1_attn_type2': opt['encoder1_attn_type2'],
        'encoder2_attn_type1': opt['encoder2_attn_type1'],
        'encoder2_attn_type2': opt['encoder2_attn_type2'],
        'encoder3_attn_type1': opt['encoder3_attn_type1'],
        'encoder3_attn_type2': opt['encoder3_attn_type2'],
        'decoder1_attn_type1': opt['decoder1_attn_type1'],
        'decoder1_attn_type2': opt['decoder1_attn_type2'],
        'decoder2_attn_type1': opt['decoder2_attn_type1'],
        'decoder2_attn_type2': opt['decoder2_attn_type2'],
        'decoder3_attn_type1': opt['decoder3_attn_type1'],
        'decoder3_attn_type2': opt['decoder3_attn_type2'],
        'encoder1_ffw_type': opt['encoder1_ffw_type'],
        'encoder2_ffw_type': opt['encoder2_ffw_type'],
        'encoder3_ffw_type': opt['encoder3_ffw_type'],
        'decoder1_ffw_type': opt['decoder1_ffw_type'],
        'decoder2_ffw_type': opt['decoder2_ffw_type'],
        'decoder3_ffw_type': opt['decoder3_ffw_type'],
        'latent_attn_type1': opt['latent_attn_type1'],
        'latent_attn_type2': opt['latent_attn_type2'],
        'latent_ffw_type': opt['latent_ffw_type'],
        'refinement_attn_type1': opt['refinement_attn_type1'],
        'refinement_attn_type2': opt['refinement_attn_type2'],
        'refinement_ffw_type': opt['refinement_ffw_type'],
        'use_both_input': opt['use_both_input'],
        'num_frames_tocache': opt.get('num_frames_tocache', 4),
        'use_dual_encoder': opt.get('use_dual_encoder', True),
        'training_mode': opt.get('training_mode', 'mixed'),
        'sensitivity': opt.get('sensitivity', 15.0)  # Default sensitivity parameter
    }
    return DeAltHDR(**model_config)


def create_video_model(opt):
    module = import_module('basicsr.models.archs.dealthdr_arch')
    model = module.make_model(opt)
    return model

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def warp(x, flow):
    """Warp image x according to flow field"""
    B, C, H, W = x.size()
    # Create coordinate grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border')
    return output

def resize_flow(flow, target_h, target_w):
    """Resize a pixel-unit flow field and scale its magnitude accordingly.
    For single-channel maps (e.g. masks), only spatial resize is applied."""
    orig_h, orig_w = flow.shape[-2:]
    if orig_h == target_h and orig_w == target_w:
        return flow
    flow_r = F.interpolate(flow, size=(target_h, target_w),
                            mode='bilinear', align_corners=False)
    if flow_r.shape[1] >= 2:           # only scale magnitude for 2-channel flow
        flow_r[:, 0] *= target_w / orig_w   # x component
        flow_r[:, 1] *= target_h / orig_h   # y component
    return flow_r


def compute_occlusion_mask(flow_fw, flow_bw, threshold=0.5):
    """Compute occlusion mask based on forward-backward consistency"""
    # Warp forward flow with backward flow
    flow_fw_warped = warp(flow_fw, flow_bw)
    # Compute consistency
    flow_diff = torch.norm(flow_fw + flow_fw_warped, dim=1, keepdim=True)
    occlusion = (flow_diff > threshold).float()
    return occlusion

class SPyNetBasicModule(nn.Module):
    """Basic module of SPyNet — 8-layer conv as in the original paper."""
    def __init__(self):
        super(SPyNetBasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    """
    SPyNet optical flow estimation network.
    Pyramid architecture from Ranjan & Black, CVPR 2017.
    Supports loading official pretrained weights via `load_pretrained`.
    """
    def __init__(self, pretrained_path=None):
        super(SPyNet, self).__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])

        # ImageNet mean/std for normalisation (as used in original SPyNet)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def _normalise(self, img):
        return (img - self.mean) / self.std

    def forward(self, ref, supp):
        """
        Estimate optical flow from ref to supp.
        Args:
            ref:  (B, C, H, W)
            supp: (B, C, H, W)
        Returns:
            flow: (B, 2, H, W)
        """
        # If single-channel features are passed, replicate to 3 channels
        if ref.shape[1] != 3:
            ref  = ref.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            supp = supp.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

        B, _, H, W = ref.shape
        ref  = self._normalise(ref.clamp(0, 1))
        supp = self._normalise(supp.clamp(0, 1))

        # Build image pyramids (6 levels, each half the previous)
        ref_pyramid  = [ref]
        supp_pyramid = [supp]
        for _ in range(5):
            ref_pyramid.append(
                F.avg_pool2d(ref_pyramid[-1], kernel_size=2, stride=2))
            supp_pyramid.append(
                F.avg_pool2d(supp_pyramid[-1], kernel_size=2, stride=2))

        flow = torch.zeros(B, 2,
                           ref_pyramid[-1].shape[2],
                           ref_pyramid[-1].shape[3],
                           device=ref.device, dtype=ref.dtype)

        for level in range(5, -1, -1):
            r = ref_pyramid[level]
            s = supp_pyramid[level]
            lh, lw = r.shape[2], r.shape[3]

            # Upsample flow from coarser level
            flow = F.interpolate(flow, size=(lh, lw),
                                 mode='bilinear', align_corners=False) * 2.0

            # Warp supp with current flow estimate
            s_warped = warp(s, flow)

            # Refine flow
            inp = torch.cat([r, s_warped, flow], dim=1)  # 3+3+2 = 8 channels
            flow = flow + self.basic_module[level](inp)

        # Upsample to original resolution
        flow = F.interpolate(flow, size=(H, W),
                             mode='bilinear', align_corners=False)
        scale_h = H / ref_pyramid[0].shape[2]
        scale_w = W / ref_pyramid[0].shape[3]
        flow[:, 0] *= scale_w
        flow[:, 1] *= scale_h
        return flow

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def clipped_softmax(tensor, dim=-1):
    # Create a mask for zero elements
    zero_mask = tensor == 0
    
    # Apply the mask to ignore zero elements in the softmax computation
    # Set zero elements to `-inf` so that they become 0 after softmax
    masked_tensor = tensor.masked_fill(zero_mask, float('-inf'))
    
    # Compute softmax on the modified tensor
    softmaxed = F.softmax(masked_tensor, dim=dim)
    
    # Zero out `-inf` elements (which are now 0 due to softmax) if any original zeros existed
    softmaxed = softmaxed.masked_fill(zero_mask, 0)
    
    non_zero_softmaxed_sum = softmaxed.sum(dim=dim, keepdim=True)
    normalized_softmaxed = softmaxed / non_zero_softmaxed_sum
    
    return normalized_softmaxed

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
# Gated Feed-Forward Network
class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GatedFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 
                                kernel_size=3, stride=1, 
                                padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
# HDR Fusion Network
class HDRFusionNet(nn.Module):
    """Fuses current frame features with up to num_neighbors FGMA-aligned
    neighbor features via learned per-pixel softmax weights."""
    def __init__(self, dim, num_neighbors=4, bias=False):
        super(HDRFusionNet, self).__init__()
        self.num_neighbors = num_neighbors
        total_in = dim * (num_neighbors + 1)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(total_in, dim * 2, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim * 2, num_neighbors + 1, kernel_size=1, bias=bias),
        )

    def forward(self, current_feat, aligned_feats):
        aligned_feats = list(aligned_feats)
        while len(aligned_feats) < self.num_neighbors:
            aligned_feats.append(torch.zeros_like(current_feat))
        all_feats = [current_feat] + aligned_feats          # N+1 tensors (B,C,H,W)
        x_cat = torch.cat(all_feats, dim=1)                 # (B, C*(N+1), H, W)
        weights = self.weight_conv(x_cat)                   # (B, N+1, H, W)
        weights = torch.softmax(weights, dim=1)
        out = torch.stack(all_feats, dim=1)                 # (B, N+1, C, H, W)
        weights = weights.unsqueeze(2)                      # (B, N+1, 1, H, W)
        return (out * weights).sum(dim=1)                   # (B, C, H, W)


# Feed_Forward Network
class FeedForward(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super(FeedForward, self).__init__()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, 
                               out_channels=ffn_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.conv4(inp)
        x = F.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return x * self.gamma

##########################################################################
## History based Attentions

class FrameHistoryRouter(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache=4):
        """
        Initializes the FrameHistoryRouter module for T-2,T-1,T+1,T+2 frame caching.
        """
        super(FrameHistoryRouter, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads= num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.num_frames_tocache = num_frames_tocache
        
    def forward(self, x, k_cached=None, v_cached=None):
        """
        Forward pass of the FrameHistoryRouter.
        Given the history states (T-2,T-1,T+1,T+2), it aggregates critical features for the restoration of the input frame
        """
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Concatenate cached key and value tensors if provided
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        
        # Calculating Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        num_cache_to_keep = int(self.num_frames_tocache*c/self.num_heads)
        return out, k[:, :, -num_cache_to_keep:, :], v[:, :, -num_cache_to_keep:, :]

class FlowGuidedMaskedAttention(nn.Module):
    """
    Flow-Guided Masked Attention (FGMA) as described in DeAltHDR paper.
    Uses optical flow for reliable regions and sparse attention for unreliable regions.
    """
    def __init__(self, dim, num_heads, bias, num_frames_tocache=4, sensitivity=15.0,
                 spynet_pretrained=None):
        super(FlowGuidedMaskedAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bias = bias
        self.num_frames_tocache = num_frames_tocache
        self.sensitivity = sensitivity  # Parameter 's' in paper

        self.spynet = SPyNet(pretrained_path=spynet_pretrained)
        
        # Separate Q, K, V projections (Projq, Projk, Projv) — paper Eq. (7)
        self.proj_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def compute_flow_mask(self, current_frame, ref_frame, sensitivity=None):
        """
        Compute binary mask using forward-backward consistency check.
        
        Args:
            current_frame: Current frame (B, C, H, W)
            ref_frame: Reference frame (B, C, H, W)
            sensitivity: Sensitivity parameter 's' (default uses self.sensitivity)
            
        Returns:
            mask: Binary mask where 1 indicates unreliable regions (B, 1, H, W)
            flow_forward: Forward optical flow
            flow_backward: Backward optical flow
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
            
        # Compute bidirectional optical flow
        flow_forward = self.spynet(current_frame, ref_frame)  # t-1 -> t
        flow_backward = self.spynet(ref_frame, current_frame)  # t -> t-1
        
        # Forward-backward consistency check (Eq. 3-4 in paper)
        warped_to_ref = warp(current_frame, flow_forward)
        warped_back = warp(warped_to_ref, flow_backward)
        
        # Compute absolute difference
        diff = torch.abs(warped_back - current_frame)
        diff_map = torch.mean(diff, dim=1, keepdim=True)  # Average over channels
        
        # Apply sensitivity threshold (Eq. 6 in paper)
        # M(i,j) = 1 if s * D(i,j) / 255 > 0.5  =>  D > 0.5 / s
        # Images are normalized to [0,1], so threshold = 0.5 / s (no factor of 255)
        threshold = 0.5 / (sensitivity + 1e-8)
        mask = (diff_map > threshold).float()
        
        return mask, flow_forward, flow_backward
        
    def forward(self, current_feat, ref_feat, sensitivity=None,
                current_img=None, ref_img=None, precomputed_flow=None):
        """
        FGMA forward pass.

        Args:
            current_feat:      Current frame features  (B, C, H, W)
            ref_feat:          Reference frame features (B, C, H, W)
            sensitivity:       Optional sensitivity override
            current_img:       Current raw image for optical flow (falls back to current_feat)
            ref_img:           Reference raw image for optical flow (falls back to ref_feat)
            precomputed_flow:  Tuple (mask, flow_fwd, flow_bwd) pre-computed at image
                               resolution. When supplied, SpyNet is skipped and the flows
                               are resized to feature resolution instead.

        Returns:
            aligned_feat: (B, 2C+1, H, W) = Concat(warped_feat, mask, attention_feat)
        """
        B, C, H, W = current_feat.shape

        if precomputed_flow is not None:
            # Reuse pre-computed flow — resize to current feature resolution
            pc_mask, pc_flow_fwd, pc_flow_bwd = precomputed_flow
            mask     = resize_flow(pc_mask.float(),     H, W)
            mask     = (mask > 0.5).float()
            flow_fwd = resize_flow(pc_flow_fwd, H, W)
            flow_bwd = resize_flow(pc_flow_bwd, H, W)
        else:
            # Compute optical flow from scratch
            flow_src = current_img if current_img is not None else current_feat
            flow_ref = ref_img     if ref_img     is not None else ref_feat
            if flow_src.shape[-2:] != (H, W):
                flow_src = F.interpolate(flow_src, size=(H, W), mode='bilinear', align_corners=False)
                flow_ref = F.interpolate(flow_ref, size=(H, W), mode='bilinear', align_corners=False)
            mask, flow_fwd, flow_bwd = self.compute_flow_mask(flow_src, flow_ref, sensitivity)

        # Step 2: Warp reference features using optical flow
        # flow_fwd = spynet(Lt, Lt-1): warp(Lt-1, flow_fwd) ≈ Lt → correct for aligning ref to current
        warped_feat = warp(ref_feat, flow_fwd)  # align ref features to current
        
        # Step 3: Compute attention ONLY in masked (unreliable) regions
        # Separate projections Projq / Projk / Projv — paper Eq. (7)
        q = self.proj_q(current_feat)
        k = self.proj_k(ref_feat)
        v = self.proj_v(ref_feat)
        
        # Apply mask to query (only compute attention in unreliable regions)
        q_masked = q * mask
        
        # Reshape for multi-head attention
        q_masked = rearrange(q_masked, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # Normalize
        q_masked = torch.nn.functional.normalize(q_masked, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        # Attention computation (Eq. 7 in paper)
        attn = (q_masked @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        # Reshape back
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        attention_feat = self.project_out(out)
        
        # Step 4: Concatenate outputs (Eq. 8 in paper)
        # Output: [warped_features, mask, attention_features]
        aligned_feat = torch.cat([warped_feat, mask, attention_feat], dim=1)
        
        return aligned_feat

##########################################################################
## Normal Attentions
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_cached=None, v_cached=None):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out, None, None

class ReducedAttn(nn.Module):
    def __init__(self, c, DW_Expand=2.0, drop_out_rate=0.):
        super().__init__()
        dw_channel = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(in_channels=c, 
                               out_channels=dw_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=dw_channel, 
                               kernel_size=3, 
                               padding=1, 
                               stride=1, 
                               groups=dw_channel,
                               bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, k_cached=None, v_cached=None):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x * self.beta, None, None

##########################################################################
class DeAltHDRBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=1, 
                 attention_type='channel', FFW_type="GFFW", num_frames_tocache=4, sensitivity=15.0):
        super(DeAltHDRBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attention_type = attention_type

        if attention_type == "Channel":
            self.attn = ChannelAttention(dim, num_heads, bias)
        elif attention_type == "ReducedAttn":
            self.attn = ReducedAttn(dim)
        elif attention_type == "FHR":  # Caches num_frames_tocache
            self.attn = FrameHistoryRouter(dim, num_heads, bias, num_frames_tocache)
        elif attention_type == "FGMA":  # Flow-Guided Masked Attention
            self.attn = FlowGuidedMaskedAttention(dim, num_heads, bias, num_frames_tocache, sensitivity)
        elif attention_type == "NoAttn":
            self.attn = None
        else:
            print(attention_type, " Not defined")
            exit()
            
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if FFW_type == "GFFW":
            self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias)
        elif FFW_type == "FFW":
            self.ffn = FeedForward(dim)
        else:
            print(FFW_type, " Not defined")
            exit()

    def forward(self, x, k_cached=None, v_cached=None, ref_feat=None, sensitivity=None):
        """
        Args:
            x: Current frame features
            k_cached, v_cached: Cached keys/values for FHR
            ref_feat: Reference frame features (for FGMA)
            sensitivity: Dynamic sensitivity parameter (for FGMA)
        """
        if self.attn is None:
            return x + self.ffn(self.norm2(x)), None, None, None
        else:
            if self.attention_type == "FGMA":
                # FGMA returns concat[warped_feat, mask, attention_feat] (2C+1 channels)
                # Fusion (HDRFusionNet) is applied OUTSIDE in DeAltHDR.forward
                if ref_feat is not None:
                    fgma_concat = self.attn(self.norm1(x), self.norm1(ref_feat), sensitivity)
                    # Extract aligned feature for residual (mask-weighted combination)
                    C = x.shape[1]
                    warped_feat    = fgma_concat[:, :C]
                    mask           = fgma_concat[:, C:C+1]
                    attention_feat = fgma_concat[:, C+1:]
                    aligned_feat   = mask * attention_feat + (1 - mask) * warped_feat
                else:
                    aligned_feat = torch.zeros_like(x)
                    fgma_concat  = None
                # residual: x stays as current feat; aligned_feat is the neighbor contribution
                x = x + aligned_feat
                x = x + self.ffn(self.norm2(x))
                # also return fgma_concat so DeAltHDR.forward can do Fusion
                return x, None, None, fgma_concat
            elif self.attention_type == "FHR":
                attn_out, k_tocache, v_tocache = self.attn(self.norm1(x), k_cached, v_cached)
                x = x + attn_out
                x = x + self.ffn(self.norm2(x))
                return x, k_tocache, v_tocache, None
            else:
                attn_out, k_tocache, v_tocache = self.attn(self.norm1(x), k_cached, v_cached)
                x = x + attn_out
                x = x + self.ffn(self.norm2(x))
                return x, k_tocache, v_tocache, None

class LevelBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                  attn_type1="Channel", attn_type2="FHR", FFW_type="GFFW", 
                  num_frames_tocache=4, num_heads=1, sensitivity=15.0):
        super(LevelBlock, self).__init__()
        self.num_blocks = num_blocks
        self.attn_type2 = attn_type2
        Block_list = []
            
        for _ in range(num_blocks - 1):
            Block_list.append(DeAltHDRBlock(dim=dim, num_heads=num_heads, 
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                             LayerNorm_type=LayerNorm_type, attention_type=attn_type1, 
                             FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                             sensitivity=sensitivity))
            
        Block_list.append(DeAltHDRBlock(dim=dim, num_heads=num_heads, 
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                        LayerNorm_type=LayerNorm_type, attention_type=attn_type2, 
                        FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                        sensitivity=sensitivity))
            
        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x, k_cached=None, v_cached=None, ref_feat=None, sensitivity=None):
        """
        Args:
            x: Current frame features
            k_cached, v_cached: Cached keys/values
            ref_feat: Reference frame features (for FGMA in last block)
            sensitivity: Dynamic sensitivity parameter
        """
        for i in range(self.num_blocks - 1):
            x, _, _, _ = self.transformer_blocks[i](x, k_cached, v_cached)

        # Pass reference features and sensitivity to the last block (if it uses FGMA)
        if self.attn_type2 == "FGMA":
            out1, k_tocache, v_tocache, fgma_concat = self.transformer_blocks[-1](
                x, k_cached, v_cached, ref_feat=ref_feat, sensitivity=sensitivity
            )
        else:
            out1, k_tocache, v_tocache, fgma_concat = self.transformer_blocks[-1](x, k_cached, v_cached)

        if k_tocache is not None:
            return out1, k_tocache, v_tocache, fgma_concat
        else:
            return out1, None, None, fgma_concat

##########################################################################
class DeAltHDR(nn.Module):
    def __init__(self,
        inp_channels,
        out_channels,
        dim,
        Enc_blocks,
        Middle_blocks,
        Dec_blocks,
        num_heads_blks, 
        num_refinement_blocks,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,

        # Encoder attention types
        encoder1_attn_type1, encoder1_attn_type2,
        encoder2_attn_type1, encoder2_attn_type2,
        encoder3_attn_type1, encoder3_attn_type2,

        # Decoder attention types
        decoder1_attn_type1, decoder1_attn_type2,
        decoder2_attn_type1, decoder2_attn_type2,
        decoder3_attn_type1, decoder3_attn_type2,

        # FFW types for each encoder and decoder level
        encoder1_ffw_type, encoder2_ffw_type, encoder3_ffw_type,
        decoder1_ffw_type, decoder2_ffw_type, decoder3_ffw_type,

        # Latent
        latent_attn_type1, latent_attn_type2, latent_ffw_type,

        # Refinement
        refinement_attn_type1, refinement_attn_type2, refinement_ffw_type,

        use_both_input,
        num_frames_tocache,
        # New parameters for dual encoder and training modes
        use_dual_encoder=True,
        training_mode='mixed',  # 'optical_flow', 'attention', 'mixed'
        sensitivity=15.0):      # Default sensitivity parameter
        super(DeAltHDR, self).__init__()
        if use_both_input:
            inp_channels *= 2
        self.use_both_input = use_both_input
        self.use_dual_encoder = use_dual_encoder
        self.training_mode = training_mode
        self.num_heads = num_heads_blks
        self.sensitivity = sensitivity
        
        # Dual encoder for long/short exposure
        if use_dual_encoder:
            # Long exposure encoder
            self.long_exposure_projection = nn.Conv2d(inp_channels, 
                                                   dim, kernel_size=3, 
                                                   stride=1, padding=1, 
                                                   bias=bias)
            # Short exposure encoder  
            self.short_exposure_projection = nn.Conv2d(inp_channels, 
                                                    dim, kernel_size=3, 
                                                    stride=1, padding=1, 
                                                    bias=bias)
        else:
            self.input_projection = nn.Conv2d(inp_channels, 
                                         dim, kernel_size=3, 
                                         stride=1, padding=1, 
                                         bias=bias) 
        
        # Encoder Levels — dual encoder: separate weights for long/short exposure
        if use_dual_encoder:
            self.long_encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                   LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                                   attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                                   FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache,
                                                   num_heads=self.num_heads[0], sensitivity=sensitivity)
            self.short_encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                    LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                                    attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                                    FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache,
                                                    num_heads=self.num_heads[0], sensitivity=sensitivity)
        else:
            self.encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                              LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                              attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                              FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache,
                                              num_heads=self.num_heads[0], sensitivity=sensitivity)

        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2

        if use_dual_encoder:
            self.long_encoder_level2 = LevelBlock(dim=int(dim*2**1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                   LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                                   attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                                   FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache,
                                                   num_heads=self.num_heads[1], sensitivity=sensitivity)
            self.short_encoder_level2 = LevelBlock(dim=int(dim*2**1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                    LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                                    attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                                    FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache,
                                                    num_heads=self.num_heads[1], sensitivity=sensitivity)
        else:
            self.encoder_level2 = LevelBlock(dim=int(dim*2**1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                              LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                              attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                              FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache,
                                              num_heads=self.num_heads[1], sensitivity=sensitivity)

        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3

        if use_dual_encoder:
            self.long_encoder_level3 = LevelBlock(dim=int(dim*2**2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                   LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                                   attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2,
                                                   FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache,
                                                   num_heads=self.num_heads[2], sensitivity=sensitivity)
            self.short_encoder_level3 = LevelBlock(dim=int(dim*2**2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                                    LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                                    attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2,
                                                    FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache,
                                                    num_heads=self.num_heads[2], sensitivity=sensitivity)
        else:
            self.encoder_level3 = LevelBlock(dim=int(dim*2**2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                              LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                              attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2,
                                              FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache,
                                              num_heads=self.num_heads[2], sensitivity=sensitivity)

        # Middle block
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = LevelBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Middle_blocks,
                                       attn_type1=latent_attn_type1, attn_type2=latent_attn_type2, 
                                       FFW_type=latent_ffw_type, 
                                       num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[3],
                                       sensitivity=sensitivity)

        # Decoder Levels
        self.up4_3 = Upsample(int(dim*2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = LevelBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[0],
                                         attn_type1=decoder1_attn_type1, attn_type2=decoder1_attn_type2, 
                                         FFW_type=decoder1_ffw_type, num_frames_tocache=num_frames_tocache, 
                                         num_heads=self.num_heads[2], sensitivity=sensitivity)
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = LevelBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[1],
                                           attn_type1=decoder2_attn_type1, attn_type2=decoder2_attn_type2, 
                                           FFW_type=decoder2_ffw_type, num_frames_tocache=num_frames_tocache, 
                                           num_heads=self.num_heads[1], sensitivity=sensitivity)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*1), kernel_size=1, bias=bias)
        self.decoder_level1 = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[2],
                                           attn_type1=decoder3_attn_type1, attn_type2=decoder3_attn_type2, 
                                           FFW_type=decoder3_ffw_type, num_frames_tocache=num_frames_tocache, 
                                           num_heads=self.num_heads[0], sensitivity=sensitivity)
        
        # Refinement Block
        self.refinement = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_refinement_blocks,
                                     attn_type1=refinement_attn_type1, attn_type2=refinement_attn_type2, 
                                     FFW_type=refinement_ffw_type, num_frames_tocache=num_frames_tocache, 
                                     num_heads=self.num_heads[0], sensitivity=sensitivity)
        
        # Standalone FGMA for each decoder level (applied once per neighbor)
        self.fgma_l3 = FlowGuidedMaskedAttention(int(dim*2**2), num_heads_blks[2], bias, num_frames_tocache, sensitivity)
        self.fgma_l2 = FlowGuidedMaskedAttention(int(dim*2**1), num_heads_blks[1], bias, num_frames_tocache, sensitivity)
        self.fgma_l1 = FlowGuidedMaskedAttention(int(dim*1),    num_heads_blks[0], bias, num_frames_tocache, sensitivity)

        # HDR Fusion: merges current + 4 aligned neighbor features (5-way)
        self.hdr_fusion_l3 = HDRFusionNet(int(dim*2**2), num_neighbors=4, bias=bias)
        self.hdr_fusion_l2 = HDRFusionNet(int(dim*2**1), num_neighbors=4, bias=bias)
        self.hdr_fusion_l1 = HDRFusionNet(int(dim*1),    num_neighbors=4, bias=bias)

        self.ending = nn.Conv2d(in_channels=int(dim*1),
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1,
                                 groups=1,
                                 bias=True)
        
        self.padder_size = (2**3)*4

    def forward(self, inp_img_, k_cached=None, v_cached=None, exposure_type=None, training_mode=None, sensitivity=None):
        """
        Forward pass with support for dual encoder and mixed training modes
        
        Args:
            inp_img_: Input frames [B, T, C, H, W] where T=5 for T-2,T-1,T,T+1,T+2
            k_cached: Cached keys (list of caches for each level)
            v_cached: Cached values (list of caches for each level)
            exposure_type: 'long' or 'short' exposure type
            training_mode: 'optical_flow', 'attention', or 'mixed'
            sensitivity: Sensitivity parameter 's' for dynamic FGMA (default: 15.0)
        """
        B, T, C, H, W = inp_img_.shape
        assert T == 5, f"Expected 5 frames (T-2,T-1,T,T+1,T+2), got {T}"

        # Gamma concatenation is done in the data pipeline (apply_hdr_preprocessing).
        # When use_both_input=True the dataset already outputs 2*n_colors channels.
        inp_img_ = self.check_image_size(inp_img_)
        
        if k_cached == None:
            k_cached = [None] * 8  # 3 encoders + 1 middle + 3 decoders + 1 refinement
            v_cached = [None] * 8

        # Extract frames: T-2, T-1, T (current), T+1, T+2
        frame_t_minus_2 = inp_img_[:, 0, :, :, :]
        frame_t_minus_1 = inp_img_[:, 1, :, :, :]
        frame_t = inp_img_[:, 2, :, :, :]  # Current frame
        frame_t_plus_1 = inp_img_[:, 3, :, :, :]
        frame_t_plus_2 = inp_img_[:, 4, :, :, :]
        
        # Determine training mode
        # Select encoder based on exposure type
        if self.use_dual_encoder and exposure_type is not None:
            if exposure_type == 'long':
                inp_enc_level1 = self.long_exposure_projection(frame_t.float())
            else:  # short exposure
                inp_enc_level1 = self.short_exposure_projection(frame_t.float())
        else:
            inp_enc_level1 = self.input_projection(frame_t.float())
        
        # Select encoder branch for current frame
        if self.use_dual_encoder and exposure_type is not None:
            enc1 = self.long_encoder_level1 if exposure_type == 'long' else self.short_encoder_level1
            enc2 = self.long_encoder_level2 if exposure_type == 'long' else self.short_encoder_level2
            enc3 = self.long_encoder_level3 if exposure_type == 'long' else self.short_encoder_level3
        else:
            enc1, enc2, enc3 = self.encoder_level1, self.encoder_level2, self.encoder_level3

        # ============ ENCODER ============
        # Level 1
        out_enc_level1, _, _, _ = enc1(inp_enc_level1, k_cached[0], v_cached[0])

        # Level 2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _, _, _ = enc2(inp_enc_level2, k_cached[1], v_cached[1])

        # Level 3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _, _, _ = enc3(inp_enc_level3, k_cached[2], v_cached[2])

        # ============ MIDDLE (LATENT) ============
        inp_latent = self.down3_4(out_enc_level3)
        latent, k_cached[3], v_cached[3], _ = self.latent(inp_latent, k_cached[3], v_cached[3])

        # ============ ENCODER: encode all 4 neighbor frames at all 3 levels ============
        neighbor_frames = [frame_t_minus_2, frame_t_minus_1, frame_t_plus_1, frame_t_plus_2]
        # Exposure parity in alternating sequence {EV-2, EV+1, EV-2, ...}:
        #   T-2 and T+2 share the SAME exposure as T; T-1 and T+1 are OPPOSITE.
        # same_as_current=True for offsets ±2, False for offsets ±1
        nb_same_exposure = [True, False, False, True]
        nb_l1_list, nb_l2_list, nb_l3_list = [], [], []
        for nb_frame, same_exp in zip(neighbor_frames, nb_same_exposure):
            if self.use_dual_encoder and exposure_type is not None:
                nb_exp = exposure_type if same_exp else ('short' if exposure_type == 'long' else 'long')
                proj   = self.long_exposure_projection  if nb_exp == 'long' else self.short_exposure_projection
                nb_enc1 = self.long_encoder_level1      if nb_exp == 'long' else self.short_encoder_level1
                nb_enc2 = self.long_encoder_level2      if nb_exp == 'long' else self.short_encoder_level2
                nb_enc3 = self.long_encoder_level3      if nb_exp == 'long' else self.short_encoder_level3
            else:
                proj = self.input_projection
                nb_enc1, nb_enc2, nb_enc3 = self.encoder_level1, self.encoder_level2, self.encoder_level3
            nb_f = proj(nb_frame.float())
            nb_l1, _, _, _ = nb_enc1(nb_f, None, None)
            nb_l2, _, _, _ = nb_enc2(self.down1_2(nb_l1), None, None)
            nb_l3, _, _, _ = nb_enc3(self.down2_3(nb_l2), None, None)
            nb_l1_list.append(nb_l1)
            nb_l2_list.append(nb_l2)
            nb_l3_list.append(nb_l3)

        def apply_fgma(fgma_mod, cur_feat, nb_feats, precomputed_flows):
            """Apply FGMA for all neighbors, reusing pre-computed flows."""
            aligned = []
            for nb_feat, pcf in zip(nb_feats, precomputed_flows):
                out = fgma_mod(cur_feat, nb_feat, precomputed_flow=pcf)
                C = cur_feat.shape[1]
                warped = out[:, :C]; mask = out[:, C:C+1]; attn = out[:, C+1:]
                aligned.append(mask * attn + (1 - mask) * warped)
            return aligned

        # ============ DECODER ============
        # Use only the first 3 (linear) channels for optical flow; exclude gamma channel
        n_flow_ch = 3
        frame_t_flow = frame_t[:, :n_flow_ch]
        neighbor_frames_flow = [f[:, :n_flow_ch] for f in neighbor_frames]

        # Pre-compute bidirectional flows for all 4 neighbors ONCE at image resolution.
        # All three FGMA decoder levels will reuse these (resize internally).
        # This reduces SpyNet calls from 24 to 8 per forward pass.
        precomputed_flows = [
            self.fgma_l3.compute_flow_mask(frame_t_flow, nb_img, sensitivity)
            for nb_img in neighbor_frames_flow
        ]  # list of (mask, flow_fwd, flow_bwd), each at image resolution

        # Decoder Level 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, k_cached[4], v_cached[4], _ = self.decoder_level3(
            inp_dec_level3, k_cached[4], v_cached[4], ref_feat=None
        )
        aligned_l3 = apply_fgma(self.fgma_l3, out_dec_level3, nb_l3_list, precomputed_flows)
        out_dec_level3 = self.hdr_fusion_l3(out_dec_level3, aligned_l3)

        # Decoder Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, k_cached[5], v_cached[5], _ = self.decoder_level2(
            inp_dec_level2, k_cached[5], v_cached[5], ref_feat=None
        )
        aligned_l2 = apply_fgma(self.fgma_l2, out_dec_level2, nb_l2_list, precomputed_flows)
        out_dec_level2 = self.hdr_fusion_l2(out_dec_level2, aligned_l2)

        # Decoder Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1, k_cached[6], v_cached[6], _ = self.decoder_level1(
            inp_dec_level1, k_cached[6], v_cached[6], ref_feat=None
        )
        aligned_l1 = apply_fgma(self.fgma_l1, out_dec_level1, nb_l1_list, precomputed_flows)
        out_dec_level1 = self.hdr_fusion_l1(out_dec_level1, aligned_l1)
        
        # ============ REFINEMENT ============
        out_refinement, k_cached[7], v_cached[7], _ = self.refinement(
            out_dec_level1, k_cached[7], v_cached[7]
        )
        
        # ============ OUTPUT ============
        out = self.ending(out_refinement)
        
        # Crop to original size (H, W extracted before padding on line 827)
        out = out[:, :, :H, :W]
        
        return out, k_cached, v_cached
    
    def check_image_size(self, x):
        _, _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from basicsr.utils.options import parse

    opt = parse(
                "options/DeAltHDR.yml",
                 is_train=True)

    model = create_video_model(opt)

    inp_shape = (5, 3, 256, 256)  # T-2,T-1,T,T+1,T+2
    macs, params = get_model_complexity_info(model, inp_shape, 
                                             verbose=False, 
                                             print_per_layer_stat=False)
    print(f"MACs: {macs}")
    print(f"Params: {params}")


