#!/usr/bin/env python3
"""Self-supervised fine-tuning of DeAltHDR on unlabeled real-world LDR videos (Sec. 3.4)."""

import argparse
import datetime
import logging
import math
import os
import random
import sys
import time
from os import path as osp
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parents[0]))

from basicsr.data import create_dataloader
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.data.real_ldr_dataset import RealLDRVideoDataset
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args, _ = parser.parse_known_args()
    opt = parse(args.opt, is_train=is_train)

    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        opt['dist_params']['timeout'] = datetime.timedelta(seconds=5400)
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    torch.manual_seed(seed + opt['rank'])
    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_dataloader(opt, logger):
    dataset_enlarge_ratio = 1
    train_set = RealLDRVideoDataset(opt, 'train')
    train_sampler = EnlargedSampler(
        train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
    train_loader = create_dataloader(
        train_set,
        opt['datasets']['train'],
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])

    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio /
        (opt['datasets']['train']['batch_size_per_gpu'] * opt['world_size']))
    total_iters = int(opt['train']['total_iter'])
    total_epochs = math.ceil(total_iters / max(num_iter_per_epoch, 1))
    logger.info(
        f'Training statistics:\n'
        f'\tReal LDR clips: {len(train_set)}\n'
        f'\tBatch size per gpu: {opt["datasets"]["train"]["batch_size_per_gpu"]}\n'
        f'\tTotal epochs: {total_epochs}; iters: {total_iters}.')
    return train_loader, train_sampler, total_epochs, total_iters


def main():
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    state_folder_path = f'experiments/{opt["name"]}/training_states/'
    try:
        states = os.listdir(state_folder_path)
    except OSError:
        states = []

    resume_state = None
    if states:
        max_state_file = f'{max(int(x[:-6]) for x in states)}.state'
        opt['path']['resume_state'] = osp.join(state_folder_path, max_state_file)

    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    logger, tb_logger = init_loggers(opt)
    train_loader, train_sampler, total_epochs, total_iters = create_train_dataloader(opt, logger)

    if resume_state:
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
        del resume_state
        torch.cuda.empty_cache()
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        raise ValueError(f'Unsupported prefetch_mode: {prefetch_mode}')

    logger.info(f'Start real fine-tuning from epoch {start_epoch}, iter {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    epoch = start_epoch

    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break

            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time

            if opt['rank'] == 0 and current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': epoch,
                    'iter': current_iter,
                    'lrs': model.get_current_learning_rate(),
                    'time': iter_time,
                    'data_time': data_time,
                }
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            if opt['rank'] == 0 and current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        epoch += 1

    consumed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of real fine-tuning. Time: {consumed}')
    if opt['rank'] == 0:
        model.save(epoch=-1, current_iter=-1)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
