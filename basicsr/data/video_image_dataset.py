from torch.utils import data as data
from basicsr.data.data_util import np2Tensor, get_patch, apply_hdr_preprocessing
from basicsr.data.transforms import random_augmentation
import os
import glob, imageio
import numpy as np
import torch

class VideoImageDataset(data.Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.name = args['name']
        self.phase = phase
        self.n_seq = args['n_sequence']
        self.n_frames_video = []
        if self.phase == "train":
            self._set_filesystem(args['dir_data'], 
                                 self.phase)
        else:
            self._set_filesystem(args['datasets']['val']['dir_data'], 
                                 self.phase)

        self.images_gt, self.images_input = self._scan()
        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        self.n_colors = args['n_colors']
        self.rgb_range = args['rgb_range']
        self.patch_size = args['patch_size']
        self.no_augment = args['no_augment']
        self.size_must_mode = args['size_must_mode']
        self.skip_corrupted = args.get('skip_corrupted', True)
        self.max_corrupted_retry = args.get('max_corrupted_retry', 100)

    def _set_filesystem(self, dir_data, phase):
        print("Loading {} => {} DataSet".format(f"{phase}", self.name))
        if isinstance(dir_data, list):
            self.dir_gt = []
            self.apath = []
            self.dir_input = []
            for path in dir_data:
                self.apath.append(path)
                self.dir_gt.append(os.path.join(path, 'gt'))
                self.dir_input.append(os.path.join(path, 'blur'))
        else:
            self.apath = dir_data
            self.dir_gt = os.path.join(self.apath, 'gt')
            self.dir_input = os.path.join(self.apath, 'blur')
        
    def _scan(self):
        if isinstance(self.dir_gt, list):
            vid_gt_names_combined = []
            vid_input_names_combined = []

            for ix in range(len(self.dir_gt)):
                vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt[ix], '*')))
                vid_input_names = sorted(glob.glob(os.path.join(self.dir_input[ix], '*')))
                
                vid_gt_names_combined.append(vid_gt_names)
                vid_input_names_combined.append(vid_input_names)
                assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"
        else:
            vid_gt_names_combined = [sorted(glob.glob(os.path.join(self.dir_gt, '*')))]
            vid_input_names_combined = [sorted(glob.glob(os.path.join(self.dir_input, '*')))]

        images_gt = []
        images_input = []
        for vid_gt, vid_input in zip(vid_gt_names_combined, vid_input_names_combined):
            for vid_gt_name, vid_input_name in zip(vid_gt, vid_input):
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))

                # Align paired sequences by the shorter side and drop too-short videos.
                paired_len = min(len(gt_dir_names), len(input_dir_names))
                if paired_len < self.n_seq:
                    print(
                        f"[WARN] Skip short/unpaired video: {vid_gt_name} | "
                        f"gt={len(gt_dir_names)}, blur={len(input_dir_names)}, required={self.n_seq}"
                    )
                    continue

                images_gt.append(gt_dir_names[:paired_len])
                images_input.append(input_dir_names[:paired_len])
                self.n_frames_video.append(paired_len)
        return images_gt, images_input

    def _load(self, images_gt, images_input):
        data_input = []
        data_gt = []
        n_videos = len(images_gt)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = np.array([imageio.imread(hr_name) for hr_name in images_gt[idx]])
            inputs = np.array([imageio.imread(lr_name) for lr_name in images_input[idx]])
            data_input.append(inputs)
            data_gt.append(gts)
        return data_gt, data_input

    def add_noise(self, x):
        # x is numpy here
        x = torch.tensor(x).unsqueeze(0).permute(0, 3, 1, 2)
        if self.phase == "train":
            # uniform sampling from [20, 50]
            r1 = 20.0/255.0
            r2 = 50.0/255.0
            stdn = np.random.rand(1,1,1,1) * (r2-r1) + r1
            stdn = torch.FloatTensor(stdn)
            noise = torch.zeros_like(x)
            noise = torch.normal(mean=noise.float(),
                                 std=stdn.expand_as(noise))
            lq = (noise + x/255.0)*255
        else:
            # in validation, the noise is fixed to 50.0/255.0.
            r2 = 50.0/255.0
            stdn = [r2]
            stdn = torch.FloatTensor(stdn)
            noise = torch.zeros_like(x)
            noise = torch.normal(mean=noise.float(),
                                 std=stdn.expand_as(noise))
            lq = (noise + x/255.0)*255

        return lq.squeeze(0).permute(1, 2, 0).numpy()

    def __getitem__(self, idx):
        retry = 0
        while True:
            try:
                inputs, gts, filenames_prompts, filenames, frame_idx = self._load_file(idx)
                if inputs.ndim != 4 or gts.ndim != 4:
                    raise ValueError(
                        f"Invalid sample shape: inputs{inputs.shape}, gts{gts.shape}, idx={idx}"
                    )
                if inputs.shape[0] != self.n_seq or gts.shape[0] != self.n_seq:
                    raise ValueError(
                        f"Invalid sequence length: inputs{inputs.shape[0]}, gts{gts.shape[0]}, expected={self.n_seq}, idx={idx}"
                    )
                break
            except Exception as err:
                if not self.skip_corrupted:
                    raise
                retry += 1
                if retry > self.max_corrupted_retry:
                    raise RuntimeError(
                        f"Exceeded max retries ({self.max_corrupted_retry}) while skipping corrupted images."
                    ) from err
                idx = (idx + 1) % self.num_frame
                print(
                    f"[WARN] Skip corrupted sample (retry {retry}/{self.max_corrupted_retry}): {err}"
                )

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat, self.size_must_mode)
        inputs_list = [inputs_concat[:, :, i*self.n_colors:(i+1)*self.n_colors] for i in range(self.n_seq)]
        gts_list = [gts_concat[:, :, i*self.n_colors:(i+1)*self.n_colors] for i in range(self.n_seq)]
        
        inputs_updated = []
        for ix in range(len(filenames_prompts)):
            _filename_ = filenames_prompts[ix]
            _img_ = inputs_list[ix]
            if "DAVIS" in _filename_:
                # denoising dataset, add noise.
                noise_added_img = self.add_noise(_img_)
                inputs_updated.append(noise_added_img)
            else:
                # let it go as is.
                inputs_updated.append(_img_)

        inputs = np.array(inputs_updated)
        gts = np.array(gts_list)

        # Apply HDR preprocessing: inverse gamma + exposure normalization
        inputs = apply_hdr_preprocessing(inputs, frame_idx)
        # inputs is now float32 in [0,1]; convert via np2Tensor with rgb_range=1
        input_tensors = [
            torch.from_numpy(np.ascontiguousarray(inputs[i].transpose(2, 0, 1))).float()
            for i in range(inputs.shape[0])
        ]
        gt_tensors = np2Tensor(*gts, rgb_range=self.rgb_range, n_colors=self.n_colors)
        
        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames_prompts, filenames

    def __len__(self):
        return self.num_frame

    def _get_index(self, idx):
        return idx % self.num_frame

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j: return i, idx
            else: idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)
        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        gts = np.array([imageio.imread(hr_name) for hr_name in f_gts])
        inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs])
        ##debug
        if inputs.ndim == 1:
            print(f"\n[DEBUG] Found dirty sequence in inputs! Index: {idx}")
            for name in f_inputs:
                print(f"  - {name}: {imageio.imread(name).shape}")
                
        if gts.ndim == 1:
            print(f"\n[DEBUG] Found dirty sequence in gts! Index: {idx}")
            for name in f_gts:
                print(f"  - {name}: {imageio.imread(name).shape}")
        ##
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]
        filenames_prompts = [x for x in f_inputs]
        return inputs, gts, filenames_prompts, filenames, frame_idx

    def _load_file_from_loaded_data(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        gts = self.data_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        inputs = self.data_input[video_idx][frame_idx:frame_idx + self.n_seq]
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]]
        print('inputs:{}'.format(inputs.shape))
        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1):
        if True:
            input, gt = get_patch(input, gt, patch_size=self.patch_size)
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]
            if not self.no_augment and self.phase == "train":
                input, gt = random_augmentation(input, gt)
        return input, gt