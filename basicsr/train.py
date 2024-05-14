import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir.replace('/basicsr',''))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str,print_metrics,
                            make_exp_dirs, mkdir_and_rename,
                           set_random_seed,Evaluator,LossTracker)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default="/data/czx/MambaCC/options/train/train_MambaIR_CC.yml", help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    return logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt, train = True)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                seed=opt['manual_seed'])
            total_epochs = int(opt['train']['total_epochs'])
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tTotal epochs: {total_epochs}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt,False)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = osp.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)

    # initialize loggers
    logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loader, total_epochs = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['epoch'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}. ")
        start_epoch = resume_state['epoch']
    else:
        model = create_model(opt)
        start_epoch = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, start_epoch)


    # training
    logger.info(
        f'Start training from epoch: {start_epoch}')
    # for epoch in range(start_epoch, total_epochs + 1):
    print("\n**************************************************************")
    print(f"\t\t Start training from epoch: {start_epoch}")
    print("**************************************************************\n")
    evaluator = Evaluator()
    best_val_loss, best_metrics = 100000.0, evaluator.get_best_metrics()
    train_loss, val_loss, ang_loss = LossTracker(), LossTracker(), LossTracker()
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss.reset()
        ang_loss.reset()
        # update learning rate
        model.update_learning_rate(epoch, warmup_epoch=opt['train'].get('warmup_epoch', -1))
        for i, train_data in enumerate(train_loader):
            # training
            model.feed_data(train_data)
            loss = model.optimize_parameters()
            train_loss.update(loss)
            print("[ Epoch: {}/{} - Batch: {} ] | [ Train loss: {:.4f} ]".format(epoch, total_epochs, i, loss))
            # log
        if epoch % opt['logger']['print_freq'] == 0:
            log_vars = {'epoch': epoch}
            log_vars.update({'lrs': model.get_current_learning_rate()})
            log_vars.update(model.get_current_log())
            msg_logger(log_vars)
            # save models and training states
        if epoch>0 and epoch % opt['logger']['save_checkpoint_freq'] == 0:
            print("Saving models and training states.")
            logger.info('Saving models and training states.')
            model.save(epoch)
        # validation
        if epoch % opt['val']['val_freq'] == 0:
            val_loss.reset()
            model.validation(val_loader, val_loss, evaluator)
        metrics = evaluator.compute_metrics()
        print("\n********************************************************************")
        print(" Train Loss ... : {:.4f}".format(train_loss.avg))
        print("....................................................................")
        print(" Val Loss ..... : {:.4f}".format(val_loss.avg))
        print("....................................................................")
        print_metrics(metrics, best_metrics)
        print("********************************************************************\n")
    # end of epoch
    logger.info('Save the latest model.')
    model.save(epoch=-1)  # -1 stands for the latest


if __name__ == '__main__':
    main()
