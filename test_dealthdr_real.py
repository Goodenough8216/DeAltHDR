#!/usr/bin/env python3
"""
Inference script for DeAltHDR on real-world unlabeled LDR video sequences.

Usage:
  python test_dealthdr_real_infer.py \
      -opt options/DeAltHDR_Real.yml \
      --ckpt experiments/DeAltHDR_real_selfsup/models/net_g_latest.pth \
      --input_dir /path/to/real/test/ldr \
      --output_dir ./output_real \
      --save_lq          

"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from basicsr.models.archs.dealthdr_arch import create_video_model
from basicsr.data.data_util import apply_hdr_preprocessing
from basicsr.utils.options import parse

_LOG_DENOM = math.log(1.0 + 5000.0)


def mu_tonemap(hdr_tensor: torch.Tensor) -> torch.Tensor:
    """T(H) = log(1+5000*H)/log(1+5000), output [0,1]."""
    return torch.log1p(5000.0 * hdr_tensor.clamp(min=0)) / _LOG_DENOM


def load_model(opt: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = create_video_model(opt)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # support both raw state_dict and BasicSR-wrapped {'params': ...}
    state = ckpt.get('params', ckpt.get('state_dict', ckpt))
    # strip leading 'module.' if saved with DataParallel
    state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Loaded checkpoint: {ckpt_path}')
    return model


def load_video_frames(video_dir: str):
    """Return sorted list of image paths in a video folder."""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    paths = sorted(
        p for p in Path(video_dir).iterdir()
        if p.suffix.lower() in exts
    )
    return [str(p) for p in paths]


def read_frame(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def prepare_window(frame_paths: list, start_frame_idx: int,
                   device: torch.device) -> torch.Tensor:
    """
    Load 5 frames, apply HDR preprocessing, return [1, 5, 6, H, W] tensor.

    start_frame_idx: absolute index of the FIRST frame in the 5-frame window.
    """
    frames = np.stack([read_frame(p) for p in frame_paths], axis=0)  # (5,H,W,C)
    frames_6ch = apply_hdr_preprocessing(frames, start_frame_idx)     # (5,H,W,6)
    # -> (1, 5, 6, H, W)
    t = torch.from_numpy(
        np.ascontiguousarray(frames_6ch.transpose(0, 3, 1, 2))
    ).float().unsqueeze(0).to(device)
    return t


def infer_video(
    model: torch.nn.Module,
    frame_paths: list,
    device: torch.device,
    training_mode: str = 'mixed',
    sensitivity: float = 15.0,
    out_dir: str = '',
    save_lq: bool = False,
    video_name: str = '',
):
    n = len(frame_paths)
    if n < 5:
        print(f'  [WARN] {video_name}: only {n} frames, need >=5, skip.')
        return

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for t in range(n):
            # Build index list for 5-frame window (edge-clamp padding)
            idxs = [max(0, min(n - 1, t + off)) for off in (-2, -1, 0, 1, 2)]
            abs_start = idxs[0]  # absolute index of first frame in window
            paths_5 = [frame_paths[i] for i in idxs]

            # Exposure type of the center frame (t)
            exposure_type = 'long' if (t % 2 == 1) else 'short'

            inp = prepare_window(paths_5, abs_start, device)  # [1,5,6,H,W]

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                out_hdr, _, _ = model(
                    inp, None, None,
                    exposure_type=exposure_type,
                    training_mode=training_mode,
                    sensitivity=sensitivity,
                )

            # Tone-map and convert to uint8
            out_tm = mu_tonemap(out_hdr.float())          # [1, 3, H, W], [0,1]
            out_np = out_tm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out_uint8 = (out_np.clip(0, 1) * 255).astype(np.uint8)

            if out_dir:
                fname = Path(frame_paths[t]).stem + '.png'
                save_path = os.path.join(out_dir, fname)
                cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR))

            if save_lq and out_dir:
                lq_raw = read_frame(frame_paths[t]).astype(np.uint8)
                lq_path = os.path.join(out_dir, Path(frame_paths[t]).stem + '_lq.png')
                cv2.imwrite(lq_path, cv2.cvtColor(lq_raw, cv2.COLOR_RGB2BGR))

    print(f'  Done: {n} frames -> {out_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='DeAltHDR real-world inference')
    p.add_argument('-opt', type=str, required=True, help='YAML config (e.g. options/DeAltHDR_Real.yml)')
    p.add_argument('--ckpt', type=str, default='',
                   help='Checkpoint path. Defaults to path.pretrain_network_g in yml.')
    p.add_argument('--input_dir', type=str, required=True,
                   help='Root folder containing video sub-folders of LDR frames.')
    p.add_argument('--output_dir', type=str, default='output_real',
                   help='Root output folder (sub-folder per video will be created).')
    p.add_argument('--training_mode', type=str, default='mixed',
                   choices=['optical_flow', 'attention', 'mixed'])
    p.add_argument('--sensitivity', type=float, default=15.0,
                   help='FGMA sensitivity (paper default: 15).')
    p.add_argument('--device', type=str, default='0',
                   help='CUDA device index, or "cpu".')
    p.add_argument('--save_lq', action='store_true',
                   help='Also copy the original LDR center frame alongside the output.')
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Config
    opt = parse(args.opt, is_train=False)

    # Checkpoint
    ckpt_path = args.ckpt or opt['path'].get('pretrain_network_g', '')
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: "{ckpt_path}". '
            'Set --ckpt or path.pretrain_network_g in the yml.'
        )

    # Model
    model = load_model(opt, ckpt_path, device)

    # Discover videos
    input_root = Path(args.input_dir)
    video_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    if not video_dirs:
        # flat folder: treat as a single video
        video_dirs = [input_root]
    print(f'Found {len(video_dirs)} video(s) under {input_root}')

    for vdir in video_dirs:
        vname = vdir.name
        frames = load_video_frames(str(vdir))
        if not frames:
            print(f'  [SKIP] {vname}: no images found.')
            continue
        print(f'Processing {vname}  ({len(frames)} frames)')
        out_dir = os.path.join(args.output_dir, vname)
        infer_video(
            model, frames, device,
            training_mode=args.training_mode,
            sensitivity=args.sensitivity,
            out_dir=out_dir,
            save_lq=args.save_lq,
            video_name=vname,
        )

    print(f'\nAll done. Results in: {args.output_dir}')


if __name__ == '__main__':
    main()
