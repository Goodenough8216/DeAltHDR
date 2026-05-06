import importlib
import math
import torch
import torchvision
from collections import OrderedDict
from copy import deepcopy
import os
from os import path as osp
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from importlib import import_module
from basicsr.models.archs.dealthdr_arch import SPyNet, warp
import basicsr.loss as loss
import numpy as np
import matplotlib.pyplot as plt
import random

_LOG_DENOM = math.log(1.0 + 5000.0)  # precomputed for mu-law tone mapping

import json

def create_video_model(opt):
    module = import_module('basicsr.models.archs.dealthdr_arch')
    model = module.make_model(opt)
    return model

metric_module = importlib.import_module('basicsr.metrics')

class DeAltHDRModel(BaseModel):
    def __init__(self, opt):
        super(DeAltHDRModel, self).__init__(opt)
        self.net_g = create_video_model(opt)
        self.net_g = self.model_to_device(self.net_g)
        self.n_sequence = opt['n_sequence']
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            print("load_model", load_path)
        if self.is_train:
            self.init_training_settings()
        self.loss = loss.L1BaseLoss()
        self.vggloss = loss.VGGPerceptualLoss(self.device)
        self.rankloss = loss.L1RankLoss()
        device = self.device
        
        # init_scale=256: prevents fp16 overflow in bias gradient accumulation.
        # mu=5000 tone mapping ≈ 587x gradient at near-zero output; bias sums over
        # batch*H*W spatial dims → default scale 65536 causes fp16 overflow (NaN grad).
        self.scaler = torch.cuda.amp.GradScaler(init_scale=256, growth_interval=200)
        self.temporal_spynet = SPyNet().to(self.device)
        self.lambda_temporal = opt.get("lambda_temporal", 0.1)
        
        # Training mode configuration
        self.training_mode = opt.get('training_mode', 'mixed')
        self.use_dual_encoder = opt.get('use_dual_encoder', True)
        self.exposure_types = ['long', 'short']  # Alternating exposure types

    def init_training_settings(self):
        self.net_g.train()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        train_opt['optim_g'].pop('type')
        self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                            **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
    def get_training_mode_for_batch(self, batch_size):
        """Determine training mode for each sample in the batch"""
        # 30% optical flow, 30% attention, 40% FGMA (mixed)
        optical_flow_count = int(0.3 * batch_size)
        attention_count = int(0.3 * batch_size)
        mixed_count = batch_size - optical_flow_count - attention_count
        
        modes = ['optical_flow'] * optical_flow_count + \
                ['attention'] * attention_count + \
                ['mixed'] * mixed_count
        
        # Shuffle the modes
        random.shuffle(modes)
        return modes
    
    def get_exposure_types_for_batch(self, batch_size):
        """Get exposure types for each sample in the batch"""
        # Alternating between long and short exposure
        types = []
        for i in range(batch_size):
            types.append(self.exposure_types[i % len(self.exposure_types)])
        return types
    
    # method to feed the data to the model.
    def feed_data(self, data):
        lq, gt, _, _ = data
        self.lq = lq.to(self.device)   # keep FP32; autocast handles precision internally
        self.gt = gt.to(self.device)

    def get_sensitivity_for_sample(self, mode):
        """
        Get sensitivity parameter based on training mode.
        Paper uses 16 sampling points: s=0, 6 points in (0,1], 6 points in (1,100), s=15, s=100, s=∞
        """
        if mode == 'optical_flow':
            return 0.0  # Pure optical flow (no attention)
        elif mode == 'attention':
            return 1000.0  # Full mask; float('inf') overflows FP16 softmax → NaN
        else:  # mixed - use FGMA with random sensitivity
            # Sample from the 16 key points (cap at 1000 to avoid FP16 overflow)
            points_0_to_1 = [i * 1.0/6 for i in range(1, 7)]  # 6 points in (0, 1]
            points_1_to_100 = [1 + i * 99.0/6 for i in range(1, 7)]  # 6 points in (1, 100)
            sample_points = [0] + points_0_to_1 + points_1_to_100 + [15, 100, 1000.0]
            return random.choice(sample_points)
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        loss_dict = OrderedDict()
        loss_dict['l_pix'] = torch.tensor(0.0, device=self.device)
        loss_dict['l_temporal'] = torch.tensor(0.0, device=self.device)

        frame_num = self.lq.shape[1]

        batch_mode = random.choice(['optical_flow', 'attention', 'mixed', 'mixed'])
        batch_sensitivity = self.get_sensitivity_for_sample(batch_mode)
        batch_exposure = self.exposure_types[current_iter % len(self.exposure_types)]

        prev_out_g = None

        for j in range(frame_num):
            target_g_images = self.gt[:, j, :, :, :]

            if j >= 2 and j < frame_num - 2:
                input_frames = torch.stack([
                    self.lq[:, j-2, :, :, :],
                    self.lq[:, j-1, :, :, :],
                    self.lq[:, j, :, :, :],
                    self.lq[:, j+1, :, :, :],
                    self.lq[:, j+2, :, :, :]
                ], dim=1)
            else:
                frames = []
                for offset in [-2, -1, 0, 1, 2]:
                    frame_idx = max(0, min(frame_num-1, j + offset))
                    frames.append(self.lq[:, frame_idx, :, :, :])
                input_frames = torch.stack(frames, dim=1)

            # Forward pass in fp16 (autocast for speed)
            with torch.cuda.amp.autocast():
                out_g, _, _ = self.net_g(
                    input_frames, None, None,
                    exposure_type=batch_exposure,
                    training_mode=batch_mode,
                    sensitivity=batch_sensitivity
                )

            # All loss computation in fp32 to prevent fp16 overflow.
            # out_g can be fp16; .float() ensures tone mapping and loss are fp32.
            out_g_fp32 = out_g.float()

            # Temporal loss: L_time = ||O_t - W(O_{t-1}, F_{t->t-1})||_1
            if j > 0 and prev_out_g is not None:
                cur_lq  = self.lq[:, j,     :, :, :].float()
                prev_lq = self.lq[:, j - 1, :, :, :].float()
                flow_ch = min(3, cur_lq.shape[1])
                with torch.no_grad():
                    flow = self.temporal_spynet(cur_lq[:, :flow_ch], prev_lq[:, :flow_ch])
                warped_prev = warp(prev_out_g.detach().float(), flow)
                loss_dict['l_temporal'] = loss_dict['l_temporal'] + self.loss(out_g_fp32, warped_prev)

            prev_out_g = out_g

            # Reconstruction loss in tone-mapped domain (Eq. 9-10), always fp32
            out_g_tm = torch.log1p(5000.0 * out_g_fp32.clamp(min=0)) / _LOG_DENOM
            tgt_tm   = torch.log1p(5000.0 * target_g_images.float().clamp(min=0)) / _LOG_DENOM
            l_pix  = self.loss(out_g_tm, tgt_tm)
            vgg_pix = self.vggloss(out_g_tm, tgt_tm)
            loss_dict['l_pix'] = loss_dict['l_pix'] + l_pix + 0.5 * vgg_pix

        loss_dict['l_pix'] = loss_dict['l_pix'] / frame_num
        loss_dict['l_temporal'] = loss_dict['l_temporal'] / max(frame_num - 1, 1)
        l_total = loss_dict['l_pix'] + self.lambda_temporal * loss_dict['l_temporal']

        # Skip batch if loss is NaN/Inf to avoid weight corruption
        if torch.isnan(l_total) or torch.isinf(l_total):
            self.optimizer_g.zero_grad()
            self.log_dict = self.reduce_loss_dict(loss_dict)
            return

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, sensitivity=15.0):
        """
        Test function with configurable sensitivity for dynamic FLOPs adjustment
        
        Args:
            sensitivity: Sensitivity parameter for FGMA (default 15.0 for balanced mode)
        """
        self.net_g.eval()
        with torch.no_grad():
            self.outputs_list = []
            self.gt_lists = []
            self.lq_lists = []
            frame_num = self.lq.shape[1]
            k_cache, v_cache = None, None
            
            for j in range(frame_num):
                target_g_images = self.gt[:, j, :, :, :]    
                
                # Prepare input frames: T-2, T-1, T, T+1, T+2
                if j >= 2 and j < frame_num - 2:
                    input_frames = torch.stack([
                        self.lq[:, j-2, :, :, :],  # T-2
                        self.lq[:, j-1, :, :, :],  # T-1
                        self.lq[:, j, :, :, :],    # T
                        self.lq[:, j+1, :, :, :],  # T+1
                        self.lq[:, j+2, :, :, :]   # T+2
                    ], dim=1)  # [B, 5, C, H, W]
                else:
                    # Handle edge cases by padding with current frame
                    frames = []
                    for offset in [-2, -1, 0, 1, 2]:
                        frame_idx = max(0, min(frame_num-1, j + offset))
                        frames.append(self.lq[:, frame_idx, :, :, :])
                    input_frames = torch.stack(frames, dim=1)
                
                # Use mixed mode for testing with specified sensitivity
                out_g, k_cache, v_cache = self.net_g(
                    input_frames.float(), 
                    k_cache, 
                    v_cache,
                    exposure_type='long',  # Default to long exposure for testing
                    training_mode='mixed',
                    sensitivity=sensitivity
                )
                
                self.outputs_list.append(out_g)
                self.gt_lists.append(target_g_images)
                self.lq_lists.append(self.lq[:, j,:, :, :])
        self.net_g.train()
    
    def non_cached_test(self):
        # proxy to the actual scores to save time.
        self.net_g.eval()
        with torch.no_grad():
            k_cache, v_cache = None, None
            pred, _, _ = self.net_g(self.lq.float(), k_cache, v_cache)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        import os
        return self.nondist_validation(dataloader, current_iter, 
                                       tb_logger, save_img, 
                                       rgb2bgr, use_image)
    
    @staticmethod
    def _tonemap(t):
        """Apply mu-law tone mapping for display (same as training loss domain)."""
        return torch.log1p(5000.0 * t.clamp(min=0)) / _LOG_DENOM

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0
        max_save = self.opt['val'].get('max_save_imgs', 20)
        # Limit validation samples for speed; full eval can be done offline
        max_val_samples = self.opt['val'].get('max_val_samples', 2000)

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue
            if max_val_samples is not None and cnt >= max_val_samples:
                break

            try:
                folder_name, img_name = val_data[len(val_data)-1][0][0].split('.')
            except Exception:
                folder_name = f'folder_{idx:04d}'
                img_name    = f'img_{idx:04d}'

            self.feed_data(val_data)
            self.test()

            for temp_i in range(len(self.outputs_list)):
                # Tone-map HDR output — same domain as training loss (mu-law, mu=5000)
                sr_tm = self._tonemap(self.outputs_list[temp_i]).clamp(0, 1)  # [1,3,H,W] float [0,1]
                gt_tm = self._tonemap(self.gt_lists[temp_i]).clamp(0, 1)

                if save_img and idx < max_save:
                    sr_img = tensor2img(sr_tm, rgb2bgr=rgb2bgr)
                    gt_img = tensor2img(gt_tm, rgb2bgr=rgb2bgr)
                    lq_img = tensor2img(self.lq_lists[temp_i][:, :3, :, :].clamp(0, 1), rgb2bgr=rgb2bgr)
                    save_dir = osp.join(self.opt['path']['visualization'], folder_name)
                    os.makedirs(save_dir, exist_ok=True)
                    imwrite(sr_img, osp.join(save_dir, f'{img_name}_frame{temp_i}_res.png'))
                    imwrite(gt_img, osp.join(save_dir, f'{img_name}_frame{temp_i}_gt.png'))
                    imwrite(lq_img, osp.join(save_dir, f'{img_name}_frame{temp_i}_lq.png'))

                if with_metrics:
                    # Compute PSNR directly on float [0,1] tone-mapped tensors to avoid
                    # the calculate_psnr max_value=1vs255 ambiguity bug with uint8 images.
                    sr_np = sr_tm.squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # HWC [0,1]
                    gt_np = gt_tm.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                    for name, opt_ in deepcopy(self.opt['val']['metrics']).items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_np, gt_np, **opt_)

                cnt += 1
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}')
        
        if rank == 0:
            pbar.close()
            
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        # pick the current frame.
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:,1,:,:,:].detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[:,1,:,:,:].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


