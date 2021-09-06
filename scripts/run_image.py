import os
from copy import deepcopy
from src.systems import image_systems
from src.utils.utils import load_json
from src.utils.setup import process_config
from src.utils.callbacks import MoCoLRScheduler
import random, torch, numpy

import pytorch_lightning as pl
import wandb

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'PretrainViewMakerSystem': image_systems.PretrainViewMakerSystem,
    'PretrainExpertSystem': image_systems.PretrainExpertSystem,
    'TransferViewMakerSystem': image_systems.TransferViewMakerSystem,
    'TransferExpertSystem': image_systems.TransferExpertSystem,
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
            See the following for more options: 
            https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html
    '''
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None
    config = process_config(args.config)
    # Only override if specified.
    if gpu_device: config.gpu_device = gpu_device
    if args.num_workers: config.data_loader_workers = args.num_workers
    seed_everything(config.seed)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    if config.optim_params.scheduler:
        lr_callback = globals()[config.optim_params.scheduler](
            initial_lr=config.optim_params.learning_rate,
            max_epochs=config.num_epochs,
            schedule=(
                int(0.6*config.num_epochs),
                int(0.8*config.num_epochs),
            ),
        )
        callbacks = [lr_callback]
    else:
        callbacks = []

    # TODO: adjust period for saving checkpoints.
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        every_n_epochs=1,
    )
    callbacks.append(ckpt_callback)
    
    wandb.init(project='image_viewmaker', entity='vm', name=config.exp_name, config=config, sync_tensorboard=True)
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=gpu_device,
         # 'ddp' is usually faster, but we use 'dp' so the negative samples 
         # for the whole batch are used for the SimCLR loss
        # distributed_backend=config.distributed_backend or 'dp',
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
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()

    # Ensure it's a string, even if from an older config
    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args, gpu_device=gpu_device)

