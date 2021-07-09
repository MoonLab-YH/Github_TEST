import argparse
import copy
import os
import os.path as osp
import time
import warnings
import pdb
import torch
import sys
sys.path.append(os.getcwd())
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
# /drive2/YH/[MX32]Active_Tracking_Project/[MyCodes]/MMdet_study/Random_RetinaNet/tools
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector, calculate_uncertainty
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.utils.active_datasets import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        default = '/drive2/YH/[MX32]Active_Tracking_Project/[MyCodes]/MMdet_study/'
                                  'Random_RetinaNet/configs/_base_/default_runtime.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        default=True,
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        default=[6],
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)
    # changes for active learning
    X_L, X_U, X_all, all_anns = get_X_L_0(cfg)
    np.save(cfg.work_dir + '/X_L_' + '0' + '.npy', X_L)
    np.save(cfg.work_dir + '/X_U_' + '0' + '.npy', X_U)

    # changes for active learning
    for cycle in cfg.cycles:
        cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
        model = build_detector(cfg.model)
        model.init_weights()
        datasets = [build_dataset(cfg.data.train)]

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        for epoch in range(cfg.epoch):
            if epoch == cfg.epoch - 1:
                cfg.lr_config.step = cfg.lr_config.step
                cfg.evaluation.interval = cfg.epoch_ratio[0]
            else:
                cfg.lr_config.step = [1000]
                cfg.evaluation.interval = 100

            if epoch == 0:
                logger.info(f'Epoch = {epoch}, First Label Set Training')
                cfg = create_X_L_file(cfg, X_L, all_anns, cycle) # reflect results of uncertainty sampling
                datasets = [build_dataset(cfg.data.train)]
                cfg.total_epochs = cfg.epoch_ratio[0]
                cfg_bak = cfg.deepcopy()
                time.sleep(2)
                train_detector(model,
                               datasets,
                               cfg,
                               distributed=distributed,
                               validate=(not args.no_validate),
                               timestamp=timestamp,
                               meta=meta)
                cfg = cfg_bak

            cfg_bak = cfg.deepcopy()
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)
            cfg = cfg_bak

        if cycle != cfg.cycles[-1]:
            # get new labeled data
            dataset_al = build_dataset(cfg.data.test)
            data_loader = build_dataloader(dataset_al, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                           dist=False, shuffle=False)
            # set random seeds
            if args.seed is not None:
                logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
                set_random_seed(args.seed, deterministic=args.deterministic)
            cfg.seed = args.seed
            meta['seed'] = args.seed
            uncertainty = calculate_uncertainty(cfg, model, data_loader, return_box=False)
            # update labeled set
            X_L, X_U = update_X_L(uncertainty, X_all, X_L, cfg.X_S_size)
            # save set and model
            np.save(cfg.work_dir + '/X_L_' + str(cycle + 1) + '.npy', X_L)
            np.save(cfg.work_dir + '/X_U_' + str(cycle + 1) + '.npy', X_U)

if __name__ == '__main__':
    main()
