#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
  python real_iqa_metrics.py \
      --input_dir output_real \
      --out_txt   output_real/iqa_scores.txt \
      --device    0
"""

import argparse
import glob
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.registry import ARCH_REGISTRY


# ---------------------------------------------------------------------------
# Minimal InferenceModel (mirrors real_metrics.py, no external pyiqa.create_metric)
# ---------------------------------------------------------------------------
class InferenceModel(torch.nn.Module):
    def __init__(self, metric_name, device=None, **kwargs):
        super().__init__()
        self.metric_name = metric_name
        self.metric_mode = DEFAULT_CONFIGS[metric_name].get('metric_mode', 'NR')
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net_opts = OrderedDict(DEFAULT_CONFIGS[metric_name]['metric_opts'])
        net_opts.update(kwargs)
        network_type = net_opts.pop('type')
        self.net = ARCH_REGISTRY.get(network_type)(**net_opts).to(self.device)
        self.net.eval()

    def to(self, device):
        self.net.to(device)
        self.device = torch.device(device)
        return self

    @torch.no_grad()
    def forward(self, img_np: np.ndarray) -> float:
        """
        img_np: HxWxC uint8 numpy array (RGB order).
        Returns scalar score.
        """
        # Normalize uint8 → [0,1] float tensor [1,C,H,W]
        t = torch.from_numpy(
            np.ascontiguousarray(img_np.astype(np.float32).transpose(2, 0, 1) / 255.0)
        ).unsqueeze(0).to(self.device)
        score = self.net(t)
        if torch.is_tensor(score):
            score = score.squeeze().mean().item()
        return float(score)


# ---------------------------------------------------------------------------
# Image collection helpers
# ---------------------------------------------------------------------------
_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def collect_images(root: str) -> list:
    """Return sorted list of all image paths under root (recursive)."""
    paths = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, '**', f'*{ext}'), recursive=True))
        paths.extend(glob.glob(os.path.join(root, '**', f'*{ext.upper()}'), recursive=True))
    # Exclude _lq.png side-by-side saves if present
    paths = [p for p in paths if not os.path.basename(p).endswith('_lq.png')]
    return sorted(set(paths))


def read_rgb_uint8(path: str) -> np.ndarray:
    """Read image as uint8 RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'Cannot read: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='CLIPIQA / MANIQA for real-world HDR reconstruction outputs (no-reference)'
    )
    p.add_argument('--input_dir', type=str, required=True,
                   help='Folder containing output PNGs (flat or nested by video).')
    p.add_argument('--out_txt', type=str, default='',
                   help='Path to write per-image + summary results. '
                        'Default: <input_dir>/iqa_scores.txt')
    p.add_argument('--device', type=str, default='0',
                   help='CUDA device index, or "cpu".')
    p.add_argument('--max_samples', type=int, default=-1,
                   help='Limit number of images (-1 = all).')
    p.add_argument('--out_json', type=str, default='',
                   help='Also write JSON summary to this path.')
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        )
    print(f'Device: {device}')

    # Collect images
    img_paths = collect_images(args.input_dir)
    if not img_paths:
        print(f'No images found under: {args.input_dir}', file=sys.stderr)
        sys.exit(1)
    if args.max_samples > 0:
        img_paths = img_paths[:args.max_samples]
    print(f'Images to evaluate: {len(img_paths)}')

    # Load IQA models
    # NOTE: first run requires internet to download ViT backbone for MANIQA.
    # Run `python real_iqa_metrics.py ... ` once with internet, weights are cached afterward.
    print('Loading CLIPIQA model...')
    model_clip = InferenceModel('clipiqa', device=device)
    print('Loading MANIQA model...')
    # 'maniqa' (koniq) needs timm ViT; fall back to 'maniqa-pipal' whose weights may already be cached.
    try:
        model_maniqa = InferenceModel('maniqa', device=device)
    except Exception as e:
        print(f'  [WARN] maniqa (koniq) failed ({e.__class__.__name__}), trying maniqa-pipal...')
        try:
            model_maniqa = InferenceModel('maniqa-pipal', device=device)
            print('  Using maniqa-pipal.')
        except Exception as e2:
            print(f'  [WARN] maniqa-pipal also failed ({e2.__class__.__name__}).')
            print('  MANIQA requires internet on first run to download ViT backbone weights.')
            print('  Re-run with internet access; weights will be cached for offline use afterward.')
            model_maniqa = None

    # Output text file
    out_txt = args.out_txt or os.path.join(args.input_dir, 'iqa_scores.txt')
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)

    scores_clip   = []
    scores_maniqa = []

    with open(out_txt, 'w', encoding='utf-8') as sf:
        sf.write(f'input_dir: {args.input_dir}\n')
        sf.write(f'{"image":<60}  clipiqa    maniqa\n')
        sf.write('-' * 80 + '\n')

        for img_path in tqdm(img_paths, desc='IQA'):
            rel = os.path.relpath(img_path, args.input_dir)
            try:
                img = read_rgb_uint8(img_path)
                s_clip = model_clip(img)
                torch.cuda.empty_cache()
                s_maniqa = model_maniqa(img) if model_maniqa is not None else float('nan')
                torch.cuda.empty_cache()
            except Exception as e:
                tqdm.write(f'[error] {rel}: {e}')
                continue

            scores_clip.append(s_clip)
            if not np.isnan(s_maniqa):
                scores_maniqa.append(s_maniqa)
            maniqa_str = f'{s_maniqa:.4f}' if not np.isnan(s_maniqa) else 'N/A'
            sf.write(f'{rel:<60}  {s_clip:.4f}     {maniqa_str}\n')

        n = len(scores_clip)
        if n == 0:
            print('No valid scores.', file=sys.stderr)
            sys.exit(1)

        avg_clip   = sum(scores_clip) / n
        avg_maniqa = sum(scores_maniqa) / len(scores_maniqa) if scores_maniqa else float('nan')

        sf.write('-' * 80 + '\n')
        maniqa_avg_str = f'{avg_maniqa:.4f}' if not np.isnan(avg_maniqa) else 'N/A (model unavailable)'
        sf.write(f'{"Average (" + str(n) + " imgs)":<60}  {avg_clip:.4f}     {maniqa_avg_str}\n')

    print(f'\n========== Results ({n} images) ==========')
    print(f'  CLIPIQA  : {avg_clip:.4f}')
    if not np.isnan(avg_maniqa):
        print(f'  MANIQA   : {avg_maniqa:.4f}')
    else:
        print(f'  MANIQA   : N/A (needs internet on first run to download ViT weights)')
    print(f'  Saved to : {out_txt}')

    if args.out_json:
        import json
        summary = {'num_images': n, 'CLIPIQA': avg_clip,
                   'MANIQA': avg_maniqa if not np.isnan(avg_maniqa) else None}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'  JSON     : {args.out_json}')


if __name__ == '__main__':
    with torch.no_grad():
        main()
