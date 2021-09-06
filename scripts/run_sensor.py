import os
import wandb
from copy import deepcopy
from src.systems import sensor_systems
from src.utils.utils import load_json
from src.utils.setup import process_config
import random
import torch
import numpy

import pytorch_lightning as pl

SYSTEM = {
    'PretrainViewMakerSystem': sensor_systems.PretrainViewMakerSystem,
    'TransferViewMakerSystem': sensor_systems.TransferViewMakerSystem,
    'PretrainExpertSimCLRSystem': sensor_systems.PretrainExpertSimCLRSystem,
    'TransferExpertSystem': sensor_systems.TransferExpertSystem,
}


def run(args, gpu_device=None):
    '''Run the Lightning system. 

    Args:
        args
            args.config_path: str, filepath to the config file
        gpu_device: str or None, specifies GPU device as follows:
            None: CPU (specified as null in config)
            'cpu': CPU
            '-1': All available GPUs
            '0': GPU 0
            '4': GPU 4
            '0,3' GPUs 1 and 3
            See: https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html
    '''
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None

    config = process_config(args.config)

    # Only override if specified.
    if gpu_device:
        config.gpu_device = gpu_device
    if args.quick:
        config.quick = args.quick
    if args.num_workers is not None:
        config.data_loader_workers = args.num_workers

    seed_everything(config.seed)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        every_n_epochs=1,
    )
    wandb.init(project='sensor_viewmaker', entity='vm',
               name=config.exp_name, config=config, sync_tensorboard=True)
    callbacks = [ckpt_callback]

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=gpu_device,
        distributed_backend=config.distributed_backend or 'dp',
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=True,
        resume_from_checkpoint=args.ckpt or config.continue_from_checkpoint,
        profiler=args.profiler,
        precision=config.optim_params.precision or 32,
        callbacks=callbacks,
        val_check_interval=config.val_check_interval or 1.0,
        limit_val_batches=config.limit_val_batches or 1.0,
    )
    trainer.fit(system)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    parser.add_argument('--gpu-device', type=str, default=None)
    parser.add_argument('--profiler', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args()

    # Ensure it's a string, even if from an older config
    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args, gpu_device=gpu_device)
