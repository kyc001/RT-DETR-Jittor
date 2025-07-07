"""by lyuwenyu
"""

import argparse
import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume,
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='pytorch_rt_detr/configs/rtdetr/rtdetr_r18vd_6x_coco.yml', help='Path to config file')
    parser.add_argument('--img_dir', type=str, default='data/coco/val2017',
                        help='Image directory for training/validation')
    parser.add_argument('--ann_file', type=str, default='data/coco/annotations/instances_val2017.json',
                        help='Annotation file for training/validation')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--num_queries', type=int, help='Number of queries')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for dataloader')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr_drop_epoch', type=int,
                        help='LR scheduler step size')
    parser.add_argument('--clip_max_norm', type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--subset_size', type=int, default=100,
                        help='Subset size for quick experiments')
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    # 构建额外参数字典，用于覆盖YAMLConfig
    override_kwargs = {}
    if args.num_classes is not None:
        override_kwargs['num_classes'] = args.num_classes
    if args.epochs is not None:
        override_kwargs['epoches'] = args.epochs
    if args.clip_max_norm is not None:
        override_kwargs['clip_max_norm'] = args.clip_max_norm
    if args.lr is not None or args.weight_decay is not None:
        override_kwargs['optimizer'] = {}
        if args.lr is not None:
            override_kwargs['optimizer']['lr'] = args.lr
        if args.weight_decay is not None:
            override_kwargs['optimizer']['weight_decay'] = args.weight_decay
    if args.lr_drop_epoch is not None:
        override_kwargs['lr_scheduler'] = {'step_size': args.lr_drop_epoch}
    # 数据相关
    if args.img_dir is not None or args.ann_file is not None or args.batch_size is not None or args.subset_size is not None:
        override_kwargs['train_dataloader'] = {
            'dataset': {}, 'batch_size': None}
        override_kwargs['val_dataloader'] = {'dataset': {}, 'batch_size': None}
        if args.img_dir is not None:
            override_kwargs['train_dataloader']['dataset']['img_folder'] = args.img_dir
            override_kwargs['val_dataloader']['dataset']['img_folder'] = args.img_dir
        if args.ann_file is not None:
            override_kwargs['train_dataloader']['dataset']['ann_file'] = args.ann_file
            override_kwargs['val_dataloader']['dataset']['ann_file'] = args.ann_file
        if args.batch_size is not None:
            override_kwargs['train_dataloader']['batch_size'] = args.batch_size
            override_kwargs['val_dataloader']['batch_size'] = args.batch_size
        if args.subset_size is not None:
            override_kwargs['train_dataloader']['dataset']['subset_size'] = args.subset_size
            override_kwargs['val_dataloader']['dataset']['subset_size'] = args.subset_size
    if args.num_queries is not None:
        override_kwargs['model'] = {'num_queries': args.num_queries}
    # 初始化main时传入
    main_arg = args
    main_arg.override_kwargs = override_kwargs

    def main_with_override(args):
        # 兼容原main
        dist.init_distributed()
        if args.seed is not None:
            dist.set_seed(args.seed)
        assert not all([args.tuning, args.resume]
                       ), 'Only support from_scrach or resume or tuning at one time'
        # 合并参数
        cfg = YAMLConfig(
            args.config,
            resume=args.resume,
            use_amp=args.amp,
            tuning=args.tuning,
            **getattr(args, 'override_kwargs', {})
        )
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        if args.test_only:
            solver.val()
        else:
            solver.fit()
    main_with_override(main_arg)
