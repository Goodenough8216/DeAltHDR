#!/usr/bin/env python3
"""
Inference script for DeAltHDR on Synthetic LDR video sequences.

Usage:
  python predict_dealthdr.py \
      -opt options/DeAltHDR.yml \
      --input_dir /hdd1/zhangshuohao/data/synthetic/test/blur \
      --output_dir ./syn_predicted
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from basicsr.models.archs.dealthdr_arch import create_video_model
from basicsr.data.data_util import apply_hdr_preprocessing
from basicsr.utils.options import parse


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


def infer_video(
    model: torch.nn.Module,
    frame_paths: list,
    device: torch.device,
    training_mode: str = 'mixed',
    out_dir: str = '',
    video_name: str = '',
):
    n = len(frame_paths)
    if n < 5:
        print(f'  [WARN] {video_name}: only {n} frames, need >=5, skip.')
        return

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for t in tqdm(range(n), desc=f'Processing {video_name}', leave=False):
            # Build index list for 5-frame window (edge-clamp padding)
            idxs = [max(0, min(n - 1, t + off)) for off in (-2, -1, 0, 1, 2)]
            abs_start = idxs[0] # Add this to track absolute start of the window
            paths_5 = [frame_paths[i] for i in idxs]

            # Read 5 frames
            frames = np.stack([read_frame(p) for p in paths_5], axis=0)  # (5,H,W,C)

            # Apply HDR Preprocessing (Linearization, exposure alignment, and concatenation)
            # The model requires 6-channel input!
            frames_6ch = apply_hdr_preprocessing(frames, start_frame_idx=abs_start) # (5,H,W,6)

            # Convert to tensor (1, 5, 6, H, W)
            inp_tensor = torch.from_numpy(
                np.ascontiguousarray(frames_6ch.transpose(0, 3, 1, 2))
            ).float().unsqueeze(0)
            
            inp_tensor = inp_tensor.to(device)

            # Alternating exposure logic: assumption for continuous frame prediction
            exposure_type = 'long' if (t % 2 == 1) else 'short'

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_hdr, _, _ = model(
                    inp_tensor,
                    exposure_type=exposure_type,
                    training_mode=training_mode
                )

            # Output is normally [1, 3, H, W] in [0, 1] range for 3-channel setup
            out_np = out_hdr.squeeze(0).float().cpu().clamp(0, 1).numpy()
            out_np = np.transpose(out_np, (1, 2, 0)) # (H, W, 3)

            # Convert to uint8 (Assuming GT synthetic target is 8-bit RGB PNG)
            out_uint8 = (out_np * 255.0).round().astype(np.uint8)

            if out_dir:
                fname = Path(frame_paths[t]).name
                save_path = os.path.join(out_dir, fname)
                cv2.imwrite(save_path, cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR))

    print(f'  Done: {n} frames -> {out_dir}')


def parse_args():
    p = argparse.ArgumentParser(description='DeAltHDR Synthetic Inference')
    p.add_argument('-opt', type=str, required=True, help='YAML config (e.g. options/DeAltHDR.yml)')
    p.add_argument('--ckpt', type=str, default='/home/wurenlong/zsh/DeAltHDR_new/experiments/DeAltHDR_dual_encoder_mixed_training/models/net_g_190000.pth',
                   help='Checkpoint path. Defaults to path.pretrain_network_g in yml.')
    p.add_argument('--input_dir', type=str, default='/hdd1/zhangshuohao/data/synthetic/test/blur',
                   help='Root folder containing scene sub-folders sequentially.')
    p.add_argument('--start_scene', type=int, default=1,
                   help='Starting scene number to process (inclusive). Default is 1.')
    p.add_argument('--end_scene', type=int, default=5,
                   help='Ending scene number to process (inclusive). Default is all.')
    p.add_argument('--output_dir', type=str, default='/hdd1/zhangshuohao/data/syn_predicted',
                   help='Root output folder (sub-folder per scene will be created).')
    p.add_argument('--training_mode', type=str, default='mixed',
                   choices=['optical_flow', 'attention', 'mixed'])
    p.add_argument('--device', type=str, default='0',
                   help='CUDA device index, or "cpu".')
    return p.parse_args()


def main():
    args = parse_args()

    # Device Setup
    if args.device.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device}')

    # Load YAML Config
    opt = parse(args.opt, is_train=False)

    # Checkpoint Path Resolution
    ckpt_path = args.ckpt or opt.get('path', {}).get('pretrain_network_g', '')
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: "{ckpt_path}".\n'
            'Please specify via --ckpt CLI argument OR path.pretrain_network_g in yaml config.'
        )

    # Initialize Model
    model = load_model(opt, ckpt_path, device)

    # Discover Scenes
    input_root = Path(args.input_dir)
    video_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    
    # Filter scenes by range
    filtered_dirs = []
    for d in video_dirs:
        try:
            scene_num = int(d.name)
            if args.start_scene <= scene_num <= args.end_scene:
                filtered_dirs.append(d)
        except ValueError:
            # If folder name is not a number, keep it (or skip it depending on preference, here we skip non-numerical if range is specified)
            pass

    if not video_dirs:
        # If it's a flat folder just containing images, process it directly
        filtered_dirs = [input_root]
        
    print(f'Found {len(filtered_dirs)} scene(s) under {input_root} in range [{args.start_scene}, {args.end_scene}]')

    for vdir in filtered_dirs:
        vname = vdir.name
        frames = load_video_frames(str(vdir))
        
        if not frames:
            print(f'  [SKIP] {vname}: no images found.')
            continue
            
        print(f'Processing {vname} ({len(frames)} frames)')
        
        out_dir = os.path.join(args.output_dir, vname)
        infer_video(
            model=model, 
            frame_paths=frames, 
            device=device,
            training_mode=args.training_mode,
            out_dir=out_dir,
            video_name=vname,
        )

    print(f'\nAll Inference Completed! Check results in: {args.output_dir}')


if __name__ == '__main__':
    main()
