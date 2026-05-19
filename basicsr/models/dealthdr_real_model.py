"""
Self-supervised real-world fine-tuning for DeAltHDR (paper Sec. 3.4, Algorithm 1).

Reference style: BracketIRE real_model.py — online network + EMA teacher,
motion-enhanced 3-frame subset vs. 5-frame pseudo targets.
"""

import math
import random
from collections import OrderedDict

import torch

from basicsr.models.base_model import BaseModel
from basicsr.models.dealthdr_model import create_video_model
import basicsr.loss as loss

_LOG_DENOM = math.log(1.0 + 5000.0)
_IDX_CENTER = 2  # center frame in {T-2, T-1, T, T+1, T+2}
_GA_OFFSETS = (2, 4, 6)   # long-exposure neighbors: t ± 2k
_GB_OFFSETS = (1, 3, 5)   # short-exposure neighbors: t ± (2k-1)


def _neighbor_pool(center_rel, offsets, window_size=5):
    """Indices inside a 5-frame window that match Algorithm 1 pools."""
    pool = []
    for off in offsets:
        for sign in (-1, 1):
            rel = center_rel + sign * off
            if 0 <= rel < window_size:
                pool.append(rel)
    return pool


def sample_motion_subset_indices(window_size=5, center_rel=_IDX_CENTER):
    """Random long- and short-exposure neighbors (motion-enhanced sampling)."""
    ga_pool = _neighbor_pool(center_rel, _GA_OFFSETS, window_size)
    gb_pool = _neighbor_pool(center_rel, _GB_OFFSETS, window_size)
    if not ga_pool or not gb_pool:
        raise ValueError('Empty GA/GB pool — need a 5-frame temporal window.')
    return random.choice(ga_pool), random.choice(gb_pool)


def build_subset_input(frames_5, idx_a, idx_b, idx_center=_IDX_CENTER):
    """
    Pack 3-frame subset into a 5-frame tensor for DeAltHDR (T must be 5).
    Unused temporal slots are filled with the center frame.
    """
    out = frames_5.clone()
    keep = {idx_center, idx_a, idx_b}
    center = frames_5[:, idx_center]
    for i in range(frames_5.shape[1]):
        if i not in keep:
            out[:, i] = center
    return out


def mu_tonemap(hdr):
    """T(H) = log(1 + mu*H) / log(1 + mu), mu = 5000."""
    return torch.log1p(5000.0 * hdr.clamp(min=0)) / _LOG_DENOM


class DeAltHDRRealModel(BaseModel):
    """Self-supervised adaptation on unlabeled real alternating-exposure LDR videos."""

    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = create_video_model(opt)
        self.net_g = self.model_to_device(self.net_g)

        self.net_g_ema = create_video_model(opt)
        self.net_g_ema = self.model_to_device(self.net_g_ema)
        for p in self.net_g_ema.parameters():
            p.requires_grad = False

        load_path = opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              opt['path'].get('strict_load_g', True),
                              param_key=opt['path'].get('param_key', 'params'))
            self.load_network(self.net_g_ema, load_path,
                              opt['path'].get('strict_load_g', True),
                              param_key=opt['path'].get('param_key', 'params'))

        self.ema_decay = opt.get('ema_decay', 0.999)
        self.beta_ema = opt.get('beta_ema', 1.0)
        self.training_mode = opt.get('training_mode', 'mixed')
        self.sensitivity = opt.get('sensitivity', 15.0)
        self.exposure_types = ['long', 'short']

        if self.is_train:
            self.net_g.train()
            self.net_g_ema.eval()
            self.init_training_settings()
            self.criterion = loss.L1BaseLoss()
            self.scaler = torch.cuda.amp.GradScaler(init_scale=256, growth_interval=200)

    def init_training_settings(self):
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = [p for p in self.net_g.parameters() if p.requires_grad]
        optim_cfg = dict(train_opt['optim_g'])
        optim_cfg.pop('type', None)
        self.optimizer_g = torch.optim.AdamW(
            [{'params': optim_params}], **optim_cfg)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        if len(data) == 5:
            lq, _, _, _, frame_idx = data
        else:
            lq, _, _, _ = data
            frame_idx = 0
        self.lq = lq.to(self.device)
        if torch.is_tensor(frame_idx):
            self.start_frame_idx = int(frame_idx.reshape(-1)[0].item()) + _IDX_CENTER
        else:
            self.start_frame_idx = int(frame_idx) + _IDX_CENTER

    def _exposure_type_for_batch(self, current_iter):
        """Match supervised DeAltHDR training: alternate long/short by iteration."""
        if self.opt.get('use_frame_exposure', False):
            return 'long' if (self.start_frame_idx % 2 == 1) else 'short'
        return self.exposure_types[current_iter % len(self.exposure_types)]

    def _forward_center(self, net, frames_5, exposure_type):
        out, _, _ = net(
            frames_5,
            None,
            None,
            exposure_type=exposure_type,
            training_mode=self.training_mode,
            sensitivity=self.sensitivity,
        )
        return out.float()

    @torch.no_grad()
    def _update_ema(self):
        decay = self.ema_decay
        online = self.get_bare_model(self.net_g)
        ema = self.get_bare_model(self.net_g_ema)
        for k, v_ema in ema.named_parameters():
            v_online = dict(online.named_parameters())[k]
            v_ema.data.mul_(decay).add_(v_online.data, alpha=1.0 - decay)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        loss_dict = OrderedDict()
        loss_dict['l_time'] = torch.tensor(0.0, device=self.device)
        loss_dict['l_ema'] = torch.tensor(0.0, device=self.device)

        input_5 = self.lq
        idx_a, idx_b = sample_motion_subset_indices()
        input_subset = build_subset_input(input_5, idx_a, idx_b)
        exposure_type = self._exposure_type_for_batch(current_iter)

        with torch.cuda.amp.autocast():
            hat_h = self._forward_center(self.net_g, input_5, exposure_type)
            tilde_h = self._forward_center(self.net_g, input_subset, exposure_type)

            with torch.no_grad():
                ema_h = self._forward_center(self.net_g_ema, input_5, exposure_type)

            hat_tm = mu_tonemap(hat_h)
            tilde_tm = mu_tonemap(tilde_h)
            ema_tm = mu_tonemap(ema_h)

            loss_time = self.criterion(tilde_tm, hat_tm.detach())
            loss_ema = self.criterion(tilde_tm, ema_tm.detach())
            l_total = loss_time + self.beta_ema * loss_ema

        if torch.isnan(l_total) or torch.isinf(l_total):
            self.optimizer_g.zero_grad()
            loss_dict['l_time'] = loss_time.detach()
            loss_dict['l_ema'] = loss_ema.detach()
            loss_dict['l_total'] = l_total.detach()
            self.log_dict = self.reduce_loss_dict(loss_dict)
            return

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        self._update_ema()

        loss_dict['l_time'] = loss_time.detach()
        loss_dict['l_ema'] = loss_ema.detach()
        loss_dict['l_total'] = l_total.detach()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, sensitivity=None):
        """Run center-frame HDR on the 5-frame clip (inference / optional val dump)."""
        sens = self.sensitivity if sensitivity is None else sensitivity
        self.net_g.eval()
        with torch.no_grad():
            out = self._forward_center(
                self.get_bare_model(self.net_g), self.lq.float(), 'long')
            self.output = out
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        c = min(3, self.lq.shape[2])
        out_dict['lq'] = self.lq[:, _IDX_CENTER, :c].detach().cpu()
        if hasattr(self, 'output'):
            out_dict['result'] = self.output[:, :c].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_g_ema, 'net_g_ema', current_iter)
        self.save_training_state(epoch, current_iter)
