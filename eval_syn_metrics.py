from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

MU = 5000.0
LOG_DENOM = math.log(1.0 + MU)
VIS_RANGE = 255.0
# VIS_RANGE = 65535.0


def mulaw_tone_map(hdr: np.ndarray) -> np.ndarray:
    x = np.maximum(hdr.astype(np.float64), 0.0)
    return np.log1p(MU * x) / LOG_DENOM


def calc_psnr_np(sr: np.ndarray, hr: np.ndarray, value_range: float) -> float:
    diff = (sr.astype(np.float32) - hr.astype(np.float32)) / value_range
    mse = float(np.power(diff, 2).mean())
    if mse <= 0:
        return float('inf')
    return -10.0 * math.log10(mse)


def calc_ssim_np(sr: np.ndarray, hr: np.ndarray, value_range: float) -> float:
    import skimage
    kw = dict(win_size=11, data_range=value_range, gaussian_weights=True)
    ver = tuple(int(x) for x in skimage.__version__.split('.')[:2])
    if ver >= (0, 19):
        kw['channel_axis'] = 2
    else:
        kw['multichannel'] = True
    return float(ssim(hr, sr, **kw))


def lpips_norm(img: np.ndarray, value_range: float, device) -> 'torch.Tensor':
    import torch
    t = img[:, :, :, np.newaxis].transpose(3, 2, 0, 1).astype(np.float32)
    t = t / (value_range / 2.0) - 1.0
    return torch.from_numpy(np.ascontiguousarray(t)).to(device)


def calc_lpips_np(sr: np.ndarray, hr: np.ndarray, loss_fn, value_range: float, device) -> float:
    with __import__('torch').no_grad():
        return float(loss_fn(lpips_norm(sr, value_range, device),
                             lpips_norm(hr, value_range, device)).detach().cpu().item())


def crop_border(img: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return img
    return img[crop:-crop, crop:-crop]


def calc_vis_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    loss_fn_lpips,
    device,
    value_range: float = VIS_RANGE,
    crop: int = 10,
) -> np.ndarray:
    pred = crop_border(pred, crop)
    gt   = crop_border(gt,   crop)
    return np.array([
        calc_psnr_np(pred, gt, value_range),
        calc_ssim_np(pred, gt, value_range),
        calc_lpips_np(pred, gt, loss_fn_lpips, value_range, device),
    ], dtype=np.float64)


def calc_mulaw_metrics(
    pred_hdr: np.ndarray,
    gt_hdr: np.ndarray,
    loss_fn_lpips,
    device,
    crop: int = 0,
) -> np.ndarray:
    pred_u = (np.clip(mulaw_tone_map(pred_hdr), 0.0, 1.0) * VIS_RANGE).astype(np.float32)
    gt_u   = (np.clip(mulaw_tone_map(gt_hdr),   0.0, 1.0) * VIS_RANGE).astype(np.float32)
    return calc_vis_metrics(pred_u, gt_u, loss_fn_lpips, device, VIS_RANGE, crop)


def _read_linear_hdr(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext in ('.hdr', '.exr'):
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f'Cannot read: {path}')
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'Cannot read: {path}')
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = img[..., ::-1]
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# HDR-VDP-2 — MATLAB version (commented out; requires MATLAB + HDR-VDP-2 toolbox)
# Usage: set env HDRVDP_ROOT=/path/to/hdrvdp2 then call calc_hdr_vdp2_matlab(...)
# ---------------------------------------------------------------------------
# def calc_hdr_vdp2_matlab(pred_hdr, gt_hdr, peak_luminance=200.0, ppd=30.0,
#                           matlab_cmd='matlab') -> float:
#     hdrvdp_root = os.environ.get('HDRVDP_ROOT', '')
#     if not hdrvdp_root:
#         raise RuntimeError('Set HDRVDP_ROOT to the HDR-VDP-2 toolbox folder.')
#     with tempfile.TemporaryDirectory() as tmp:
#         tmp = Path(tmp)
#         pred_path = tmp / 'pred.hdr'; gt_path = tmp / 'gt.hdr'
#         out_path  = tmp / 'q.txt';   script_path = tmp / 'eval_vdp.m'
#         cv2.imwrite(str(pred_path), cv2.cvtColor(pred_hdr, cv2.COLOR_RGB2BGR))
#         cv2.imwrite(str(gt_path),   cv2.cvtColor(gt_hdr,   cv2.COLOR_RGB2BGR))
#         script = (
#             f"addpath(genpath('{hdrvdp_root}'));\n"
#             f"pred_rgb=hdrread('{pred_path}'); ref_rgb=hdrread('{gt_path}');\n"
#             f"pred_lum=({peak_luminance}).*(0.2126*pred_rgb(:,:,1)+0.7152*pred_rgb(:,:,2)+0.0722*pred_rgb(:,:,3));\n"
#             f"ref_lum =({peak_luminance}).*(0.2126*ref_rgb(:,:,1) +0.7152*ref_rgb(:,:,2) +0.0722*ref_rgb(:,:,3));\n"
#             f"res=hdrvdp(pred_lum,ref_lum,'luminance',{ppd},{{}});\n"
#             f"fid=fopen('{out_path}','w'); fprintf(fid,'%.8f\\n',res.Q); fclose(fid);\n"
#         )
#         script_path.write_text(script)
#         r = subprocess.run([matlab_cmd,'-nodisplay','-nosplash','-batch',
#                             f"run('{script_path}');"], check=True,
#                            capture_output=True, text=True)
#         if not out_path.is_file():
#             raise RuntimeError(f'MATLAB no output.\n{r.stdout}\n{r.stderr}')
#         return float(out_path.read_text().strip())


def calc_hdr_vdp2(
    pred_hdr: np.ndarray,
    gt_hdr: np.ndarray,
    peak_luminance: float = 200.0,
    ppd: float = 30.0,
) -> float:
    """
    Python implementation of HDR-VDP-2 Q score (no MATLAB required).
    Ref: Mantiuk et al., HDR-VDP-2, ACM TOG 2011.

    pred_hdr / gt_hdr : float32 HWC linear RGB, relative [0, 1].
    peak_luminance    : display peak luminance in cd/m^2.
    ppd               : pixels per degree.
    Returns Q in [0, 100], higher is better.
    """
    # Spatial frequency bands (cpd) and Gaussian sigma in pixels for each band.
    # Six bands: ~16, 8, 4, 2, 1, 0.5 cpd.
    band_freqs = np.array([16.0, 8.0, 4.0, 2.0, 1.0, 0.5])
    band_sigmas = ppd / (2.0 * band_freqs)       # sigma in pixels per band

    def rgb2lum(img: np.ndarray) -> np.ndarray:
        return np.maximum(
            0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2],
            1e-4,
        )

    # Absolute luminance (cd/m^2)
    L_t = rgb2lum(pred_hdr.astype(np.float64)) * peak_luminance
    L_r = rgb2lum(gt_hdr.astype(np.float64))  * peak_luminance

    # Log10 luminance (Weber-Fechner law)
    log_t = np.log10(L_t)
    log_r = np.log10(L_r)

    # ---- CSF model: simplified Barten (1999) --------------------------------
    # S(f, L) — contrast sensitivity at spatial frequency f (cpd), mean lum L (cd/m^2)
    def barten_csf(f: float, L: float) -> float:
        a = 440.0 * (1.0 + 0.7 / max(L, 0.1)) ** (-0.2)
        b = 0.3  * (1.0 + 100.0 / max(L, 0.1)) ** 0.15
        return float(a * f * np.exp(-b * f) * np.sqrt(1.0 + 0.06 * np.exp(b * f)))

    # ---- Band decomposition via Laplacian pyramid ----------------------------
    # Build Gaussian pyramid for test and reference images.
    def gauss_pyr(img: np.ndarray, sigmas: np.ndarray):
        pyr = [img]
        for s in sigmas:
            pyr.append(gaussian_filter(pyr[-1], sigma=max(s, 0.5)))
        return pyr

    pyr_t = gauss_pyr(log_t, band_sigmas)
    pyr_r = gauss_pyr(log_r, band_sigmas)

    # ---- Detection probability per band -------------------------------------
    # JND (just-noticeable difference in log-luminance) = 1 / CSF
    # P(detect) = 0.5*(1 + erf(|err| / (sqrt(2)*sigma_jnd)))
    # with internal noise sigma_noise added in quadrature.
    SIGMA_NOISE = 0.05   # internal noise in log-luminance units
    MINKOWSKI_P = 3.0    # pooling exponent

    lum_mean = float(np.mean((L_t + L_r) / 2.0))
    pooled = 0.0
    weight_sum = 0.0

    for i, freq in enumerate(band_freqs):
        # Band = coarser - finer level
        band_t = pyr_t[i] - pyr_t[i + 1]
        band_r = pyr_r[i] - pyr_r[i + 1]
        diff   = band_t - band_r

        csf_val = barten_csf(freq, lum_mean)
        jnd     = 1.0 / max(csf_val, 1e-6)
        sigma   = np.sqrt(jnd ** 2 + SIGMA_NOISE ** 2)

        p_det = 0.5 * (1.0 + erf(np.abs(diff) / (np.sqrt(2.0) * sigma)))

        # Weight bands by CSF * frequency (area under CSF curve)
        w = csf_val * freq
        pooled     += w * float(np.mean(p_det ** MINKOWSKI_P)) ** (1.0 / MINKOWSKI_P)
        weight_sum += w

    mean_p = pooled / max(weight_sum, 1e-8)

    # Q = 100 when no difference detected (mean_p = 0.5), 0 when fully detected (mean_p = 1).
    Q = 100.0 * (1.0 - 2.0 * max(float(mean_p) - 0.5, 0.0))
    return float(np.clip(Q, 0.0, 100.0))


def iter_syn_test(gt_dir: str, pred_dir: str, start_scene: int, end_scene: int):
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    
    if not gt_root.is_dir():
        raise FileNotFoundError(f'GT folder not found: {gt_root}')
        
    for scene_path in sorted(gt_root.iterdir()):
        if not scene_path.is_dir():
            continue
            
        try:
            scene_num = int(scene_path.name)
            if not (start_scene <= scene_num <= end_scene):
                continue
        except ValueError:
            pass # skip non-numerical folders if any
            
        scene = scene_path.name
        
        for gt_file in sorted(scene_path.glob("gt_*.png")):
            # gt_file is like gt_100.png, we need to find ldr_100.png
            frame_id = gt_file.stem.split('_')[1]
            pred_name = f'ldr_{frame_id}.png'
            pred_path = pred_root / scene / pred_name
            
            if pred_path.is_file():
                yield str(pred_path), str(gt_file), f'{scene}/{frame_id}'
            else:
                tqdm.write(f'[skip] missing pred: {pred_path}')


def iter_pair_list(pair_file: str):
    with open(pair_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                yield parts[0], parts[1], parts[0]


def read_vis_pair(pred_path: str, gt_path: str) -> tuple[np.ndarray, np.ndarray]:
    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt   = cv2.imread(gt_path,   cv2.IMREAD_UNCHANGED)
    if pred is None or gt is None:
        raise IOError(f'Failed to read {pred_path} or {gt_path}')
    if pred.ndim == 3:
        pred = pred[..., ::-1]
    if gt.ndim == 3:
        gt = gt[..., ::-1]
    return pred, gt


def resolve_hdr_paths(gt_vis_path: str, pred_vis_path: str, hdr_gt_name: str, hdr_pred_name: str):
    gt_dir    = os.path.dirname(gt_vis_path)
    pred_dir  = os.path.dirname(pred_vis_path)
    stem      = Path(pred_vis_path).stem
    cand_gt   = os.path.join(gt_dir, hdr_gt_name)
    cand_pred = os.path.join(pred_dir, hdr_pred_name) if hdr_pred_name else os.path.join(pred_dir, stem + '.hdr')
    return cand_pred, cand_gt


def parse_args():
    p = argparse.ArgumentParser(description='DeAltHDR evaluation: PSNR / SSIM / LPIPS / HDR-VDP-2')
    p.add_argument('--gt_dir',  type=str, default='/hdd1/zhangshuohao/data/synthetic/test/gt', help='GT images root folder')
    p.add_argument('--pred_dir',  type=str, default='/hdd1/zhangshuohao/data/syn_predicted', help='Predicted vis PNG folder')
    p.add_argument('--start_scene', type=int, default=1, help='Starting scene number to evaluate')
    p.add_argument('--end_scene', type=int, default=5, help='Ending scene number to evaluate')
    p.add_argument('--layout',    type=str, default='syn_test', choices=['syn_test', 'pair_list'])
    p.add_argument('--pair_file', type=str, default='', help='Two-column text file: pred_path gt_path')
    p.add_argument('--domain',    type=str, default='vis_uint16', choices=['vis_uint16', 'mulaw'],
                   help='vis_uint16: compare tone-mapped vis PNGs (range 65535); mulaw: compare after mu-law on linear HDR')
    p.add_argument('--gt_name',   type=str, default='rgb_vis_gt.png')
    p.add_argument('--crop',      type=int, default=10, help='Border pixels to crop')
    p.add_argument('--plus',      action='store_true', help='Use 16px crop instead of 10px')
    p.add_argument('--device',    type=str, default='6', help='CUDA device index for LPIPS')
    p.add_argument('--max_samples', type=int, default=-1, help='Max pairs to evaluate (-1 = all)')
    p.add_argument('--hdr_vdp',   action='store_true', help='Also compute HDR-VDP-2 Q score (Python impl, no MATLAB needed)')
    p.add_argument('--hdr_gt_name',   type=str, default='rgb_gt.hdr')
    p.add_argument('--hdr_pred_name', type=str, default='')
    p.add_argument('--peak_luminance', type=float, default=200.0, help='Peak luminance cd/m^2 for HDR-VDP-2')
    p.add_argument('--ppd',        type=float, default=30.0, help='Pixels per degree for HDR-VDP-2')
    p.add_argument('--out_json',  type=str, default='', help='Write JSON summary to this path')
    p.add_argument('--log_file',  type=str, default='', help='Append text summary to this file')
    return p.parse_args()


def main():
    args = parse_args()
    crop = 16 if args.plus else args.crop

    if args.layout == 'syn_test':
        if not args.gt_dir or not args.pred_dir:
            print('syn_test requires --gt_dir and --pred_dir', file=sys.stderr)
            sys.exit(1)
        pairs = list(iter_syn_test(args.gt_dir, args.pred_dir, args.start_scene, args.end_scene))
    else:
        if not args.pair_file:
            print('pair_list requires --pair_file', file=sys.stderr)
            sys.exit(1)
        pairs = list(iter_pair_list(args.pair_file))

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    if not pairs:
        print('No valid pairs found.', file=sys.stderr)
        sys.exit(1)

    import torch
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
    except ImportError:
        print('lpips not installed: pip install lpips', file=sys.stderr)
        sys.exit(1)

    n = len(pairs)
    metrics    = np.full((n, 3), np.nan, dtype=np.float64)
    vdp_scores = []

    print(f'domain={args.domain}  pairs={n}  crop={crop}px')

    for i, (pred_path, gt_path, key) in enumerate(tqdm(pairs, desc='eval')):
        try:
            if args.domain == 'vis_uint16':
                pred, gt = read_vis_pair(pred_path, gt_path)
                metrics[i] = calc_vis_metrics(pred, gt, loss_fn, device, VIS_RANGE, crop)
            else:
                pred = _read_linear_hdr(pred_path)
                gt   = _read_linear_hdr(gt_path)
                metrics[i] = calc_mulaw_metrics(pred, gt, loss_fn, device, crop=crop)

            if args.hdr_vdp:
                hp, hg = resolve_hdr_paths(gt_path, pred_path, args.hdr_gt_name, args.hdr_pred_name)
                if os.path.isfile(hp) and os.path.isfile(hg):
                    ph = _read_linear_hdr(hp)
                    gh = _read_linear_hdr(hg)
                    vdp_scores.append(calc_hdr_vdp2(ph, gh, args.peak_luminance, args.ppd))
        except Exception as e:
            tqdm.write(f'[error] {key}: {e}')

    # PSNR can be +inf for identical pred/gt; do not drop those rows.
    valid_mask = (
        (np.isfinite(metrics[:, 0]) | np.isposinf(metrics[:, 0]))
        & np.isfinite(metrics[:, 1])
        & np.isfinite(metrics[:, 2])
    )
    valid = metrics[valid_mask]
    if len(valid) == 0:
        print('No valid metric rows.', file=sys.stderr)
        sys.exit(1)

    psnr_col = valid[:, 0].copy()
    psnr_col[np.isposinf(psnr_col)] = 100.0  # cap for mean reporting
    mean_m = np.array([psnr_col.mean(), valid[:, 1].mean(), valid[:, 2].mean()])
    summary = {
        'domain':    args.domain,
        'num_pairs': int(len(valid)),
        'PSNR':      float(mean_m[0]),
        'SSIM':      float(mean_m[1]),
        'LPIPS':     float(mean_m[2]),
    }
    if vdp_scores:
        summary['HDR-VDP-2'] = float(np.mean(vdp_scores))
        summary['HDR-VDP-2_n'] = len(vdp_scores)

    print('\n========== Results ==========')
    print(f'  PSNR      : {summary["PSNR"]:.4f} dB')
    print(f'  SSIM      : {summary["SSIM"]:.6f}')
    print(f'  LPIPS     : {summary["LPIPS"]:.6f}')
    if vdp_scores:
        print(f'  HDR-VDP-2 : {summary["HDR-VDP-2"]:.4f}  (n={len(vdp_scores)})')
    elif args.hdr_vdp:
        print('  HDR-VDP-2 : skipped (linear HDR files not found or MATLAB unavailable)')

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'Wrote {args.out_json}')

    if args.log_file:
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(f'\npred_dir: {args.pred_dir}\n')
            f.write(f'  PSNR={summary["PSNR"]:.2f} SSIM={summary["SSIM"]:.4f} LPIPS={summary["LPIPS"]:.3f}\n')
            if vdp_scores:
                f.write(f'  HDR-VDP-2={summary["HDR-VDP-2"]:.4f}\n')


if __name__ == '__main__':
    main()
