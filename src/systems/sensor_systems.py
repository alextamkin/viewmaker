"""
Contrastive learning on wearable sensor data

Contains PL Systems for SimCLR and Viewmaker methods (pretraining and linear evaluation).
"""

import os
import math
from dotmap import DotMap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.datasets.pamap2 import PAMAP2

from src.models.transfer import LogisticRegression
from src.models import resnet_small
from src.objectives.memory_bank import MemoryBank

from src.utils.utils import l2_normalize, frozen_params, load_json
from src.systems.image_systems import create_dataloader
from src.objectives.simclr import SimCLRObjective
from src.objectives.adversarial import AdversarialSimCLRLoss
from src.objectives.infonce import NoiseConstrastiveEstimation

from src.models import viewmaker
import pytorch_lightning as pl
import wandb


class PretrainExpertInstDiscSystem(pl.LightningModule):
    '''Pretraining with Instance Discrimination
    
    NOTE: only the SimCLR model was used for PAMAP2 in the paper.
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_encoder()
        self.memory_bank = MemoryBank(len(self.train_dataset), 
                                      self.config.model_params.out_dim) 
        self.memory_bank_labels = MemoryBank(len(self.train_dataset), 1, dtype=int)

    def create_datasets(self):
        print('Initializing validation dataset.')
        # We use a larger default value of 50k examples for validation to reduce variance.
        val_dataset = PAMAP2(
            mode='val',
            examples_per_epoch=self.config.data_params.val_examples_per_epoch or 50000
        )

        if not self.config.quick:
            print('Initializing train dataset.')
            train_dataset = PAMAP2(
                mode='train',
                examples_per_epoch=self.config.data_params.train_examples_per_epoch or 10000
            )
            if not self.config.data_params.train_examples_per_epoch:
                print('WARNING: self.config.data_params.train_examples_per_epoch not specified. Using default value of 10k')
        else:
            train_dataset = val_dataset
        return train_dataset, val_dataset

    def create_encoder(self):
        if self.config.model_params.resnet_small:
            encoder_model = resnet_small.ResNet18(
                self.config.model_params.out_dim,
                num_channels=52, # 52 feature spectrograms
                input_size=32,
            )
        else:
            resnet_class = getattr(
                torchvision.models, 
                self.config.model_params.resnet_version,
            )
            encoder_model = resnet_class(
                pretrained=False,
                num_classes=self.config.model_params.out_dim,
            )
            encoder_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                            padding=3, bias=False)

        if self.config.model_params.projection_head:
            mlp_dim = encoder_model.linear.weight.size(1)
            encoder_model.linear = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                encoder_model.linear,
            )
        return encoder_model

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(),
                                lr=self.config.optim_params.learning_rate,
                                momentum=self.config.optim_params.momentum,
                                weight_decay=self.config.optim_params.weight_decay)
        return [optim], []

    def forward(self, inputs):
        return self.model(inputs)

    def get_losses_for_batch(self, batch):
        indices, inputs1, inputs2, _ = batch
        outputs = self.forward(inputs1)
        loss_fn = NoiseConstrastiveEstimation(indices, outputs, self.memory_bank,
                                              k=self.config.loss_params.k,
                                              t=self.config.loss_params.t,
                                              m=self.config.loss_params.m)
        loss = loss_fn.get_loss()

        with torch.no_grad():
            new_data_memory = loss_fn.updated_new_data_memory()
            self.memory_bank.update(indices, new_data_memory)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        metrics = {'loss': loss}
        return {'loss': loss, 'log': metrics}

    def get_nearest_neighbor_label(self, embs, labels):
        """
        NOTE: ONLY TO BE USED FOR VALIDATION.
        
        For each example in validation, find the nearest example in the 
        training dataset using the memory bank. Assume its label as
        the predicted label.
        """
        all_dps = self.memory_bank.get_all_dot_products(embs)
        _, neighbor_idxs = torch.topk(all_dps, k=1, sorted=False, dim=1)
        neighbor_idxs = neighbor_idxs.squeeze(1)
        
        neighbor_labels = self.memory_bank_labels.at_idxs(neighbor_idxs).squeeze(-1)
        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()

        return num_correct, embs.size(0)
    
    def validation_step(self, batch, batch_idx):
        _, inputs1, inputs2, labels = batch
        outputs = self.model(inputs1)
        num_correct, batch_size = self.get_nearest_neighbor_label(outputs, labels)
        num_correct = torch.tensor(num_correct, dtype=float, device=self.device)
        batch_size = torch.tensor(batch_size, dtype=float, device=self.device)
        return OrderedDict({'val_num_correct': num_correct,
                            'val_num_total': batch_size})

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.stack([elem[key] for elem in outputs]).mean()
        num_correct = torch.stack([out['val_num_correct'] for out in outputs]).sum()
        num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        progress_bar = {'acc': val_acc}
        return {'log': metrics, 'val_acc': val_acc, 'progress_bar': progress_bar}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)
    
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)


class PretrainExpertSimCLRSystem(PretrainExpertInstDiscSystem):
    '''SimCLR Pretraining for PAMAP2'''

    def create_datasets(self):
        print('Initializing validation dataset.')
        val_dataset = PAMAP2(
            mode='val',
            examples_per_epoch=self.config.data_params.val_examples_per_epoch or 50000,
            sensor_transforms=None,
         )

        if not self.config.quick:
            print('Initializing train dataset.')
            train_dataset = PAMAP2(
                mode='train',
                examples_per_epoch=self.config.data_params.train_examples_per_epoch or 10000,
                sensor_transforms=self.config.data_params.sensor_transforms
            )
            if not self.config.data_params.train_examples_per_epoch:
                print('WARNING: self.config.data_params.train_examples_per_epoch not specified. Using default value of 10k')
        else:
            train_dataset = val_dataset
        return train_dataset, val_dataset

    def get_losses_for_batch(self, batch):
        indices, inputs1, inputs2, labels = batch
        outputs1 = self.forward(inputs1)
        outputs2 = self.forward(inputs2)
        loss_fn = SimCLRObjective(outputs1, outputs2, 
                                  t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        with torch.no_grad():  # for nearest neighbor
            new_data_memory = (l2_normalize(outputs1, dim=1) + 
                               l2_normalize(outputs2, dim=1)) / 2.
            self.memory_bank.update(indices, new_data_memory)
            self.memory_bank_labels.update(indices, labels.unsqueeze(1))

        return loss


class PretrainViewMakerSystem(PretrainExpertSimCLRSystem):
    """
    SimCLR + Viewmaker Pretraining.
    """

    def __init__(self, config):
        super().__init__(config)
        self.view = self.create_viewmaker()

    def create_datasets(self):
        print('Initializing validation dataset.')
        val_dataset = PAMAP2(
            mode='val',
            examples_per_epoch=self.config.data_params.val_examples_per_epoch or 50000
        )

        if self.config.quick:
            train_dataset = val_dataset
        else:
            print('Initializing train dataset.')
            train_dataset = PAMAP2(
                mode='train',
                examples_per_epoch=self.config.data_params.train_examples_per_epoch or 10000
            )
            if not self.config.data_params.train_examples_per_epoch:
                print(
                    'WARNING: self.config.data_params.train_examples_per_epoch not specified. Using default value of 10k')
        return train_dataset, val_dataset

    def create_viewmaker(self):
        view_model = viewmaker.Viewmaker(
            num_channels=52,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            clamp=False,
        )
        return view_model

    def configure_optimizers(self):
        encoder_optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum,
            weight_decay=self.config.optim_params.weight_decay,
        )
        view_optim_name = self.config.optim_params.viewmaker_optim
        view_parameters = self.view.parameters()
        if view_optim_name == 'adam':
            view_optim = torch.optim.Adam(view_parameters)
        elif not view_optim_name or view_optim_name == 'sgd':
            view_optim = torch.optim.SGD(
                view_parameters,
                lr=self.config.optim_params.viewmaker_learning_rate or self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        else:
            raise ValueError(f'Optimizer {view_optim_name} not implemented')

        return [encoder_optim, view_optim], []

    def forward(self, batch):
        indices, inputs, inputs2, labels = batch
        view1 = self.view(inputs)
        view2 = self.view(inputs2)

        if self.config.model_params.view_clip:
            num_std = self.config.model_params.view_clip_num_std
            tot_std = num_std * self.train_dataset.normalize_stdev
            view_min = self.train_dataset.normalize_mean - tot_std
            view_max = self.train_dataset.normalize_mean + tot_std
            view1 = torch.clamp(view1, view_min, view_max) 
            view2 = torch.clamp(view2, view_min, view_max) 

        emb_dict = {
            'indices': indices,
            'view1_embs': self.model(view1),
            'view2_embs': self.model(view2),
            'labels': labels,
        }

        if self.global_step % (len(self.train_dataset) // self.batch_size) == 0:
            views_to_log = view1.detach()[0].view(-1,32,32,1).cpu().numpy()
            wandb.log({"examples": [wandb.Image(view, caption=f"Epoch: {self.current_epoch}, Step {self.global_step}") for view in views_to_log]})

        return emb_dict

    def get_losses_for_batch(self, emb_dict):
        loss_function = AdversarialSimCLRLoss(
            embs1=emb_dict['view1_embs'],
            embs2=emb_dict['view2_embs'],
            t=self.config.loss_params.t,
            view_maker_loss_weight=self.config.loss_params.view_maker_loss_weight
        )
        encoder_loss, view_maker_loss = loss_function.get_loss()
        
        with torch.no_grad():
            new_data_memory = l2_normalize(emb_dict['view1_embs'].detach(), dim=1)
            self.memory_bank.update(emb_dict['indices'], new_data_memory)
            self.memory_bank_labels.update(emb_dict['indices'], emb_dict['labels'].unsqueeze(1))

        return encoder_loss, view_maker_loss

    def get_view_bound_magnitude(self):
        return self.config.model_params.view_bound_magnitude

    def training_step(self, batch, batch_idx, optimizer_idx):
        emb_dict = self.forward(batch)
        emb_dict['optimizer_idx'] = torch.tensor(optimizer_idx, device=self.device)
        return emb_dict
    
    def training_step_end(self, emb_dict):
        encoder_loss, view_maker_loss = self.get_losses_for_batch(emb_dict)

        # Handle Tensor (dp) and int (ddp) cases
        if emb_dict['optimizer_idx'].__class__ == int or emb_dict['optimizer_idx'].dim() == 0:
            optimizer_idx = emb_dict['optimizer_idx'] 
        else:
            optimizer_idx = emb_dict['optimizer_idx'][0]

        if optimizer_idx == 0:
            metrics = {
                'encoder_loss': encoder_loss,
            }
            return {'loss': encoder_loss, 'log': metrics}
        else:
            # update the bound allowed for view
            self.view.bound_magnitude = self.get_view_bound_magnitude()

            metrics = {'view_maker_loss': view_maker_loss}
            return {'loss': view_maker_loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        _, inputs1, inputs2, labels = batch
        outputs = self.model(inputs1)
        num_correct, batch_size = self.get_nearest_neighbor_label(outputs, labels)
        output = OrderedDict({
            'val_num_correct': torch.tensor(num_correct, dtype=float, device=self.device),
            'val_num_total': torch.tensor(batch_size, dtype=float, device=self.device),
        })
        return output

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.stack([elem[key] for elem in outputs]).mean()
        num_correct = torch.stack([out['val_num_correct'] for out in outputs]).sum()
        num_total = torch.stack([out['val_num_total'] for out in outputs]).sum()
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        progress_bar = {'acc': val_acc}
        return {'log': metrics, 'val_acc': val_acc, 'progress_bar': progress_bar}


class TransferViewMakerSystem(pl.LightningModule):
    '''Linear Evaluation for Viewmaker + SimCLR models.'''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size

        self.encoder, self.viewmaker, self.system, self.pretrain_config = self.load_pretrained_model()
        resnet = self.pretrain_config.model_params.resnet_version
        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.pretrain_config.model_params.resnet_small:
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 2 * 2
            else:
                num_features = 512
        elif resnet == 'resnet50':
            num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')

        if not self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                cut_ix = -2
            else:
                cut_ix = -1
            self.encoder = nn.Sequential(*list(self.encoder.children())[:cut_ix])

        if not self.config.optim_params.supervised:
            self.encoder = self.encoder.eval()
            frozen_params(self.encoder)

        frozen_params(self.viewmaker)

        self.num_features = num_features
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_model()

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = load_json(config_path)
        config = DotMap(config_json)

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        # If supervised, train from scratch. 
        if not self.config.optim_params.supervised:
            system.load_state_dict(checkpoint['state_dict'])
        
        viewmaker = system.view.eval()
        encoder = system.model

        if not self.config.optim_params.supervised:
            encoder = system.model.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        for param in viewmaker.parameters():
            param.requires_grad = False

        return encoder, viewmaker, system, system.config

    def create_datasets(self):
        print('Initializing train dataset.')
        if self.config.data_params.train_small:
            train_mode = 'train_small'
        else:
            train_mode = 'train'
        train_dataset = PAMAP2(
            mode=train_mode,
            examples_per_epoch=self.pretrain_config.data_params.train_examples_per_epoch or 10000
        )
        print('Initializing validation dataset.')
        val_dataset = PAMAP2(
            mode='val',
            examples_per_epoch=self.pretrain_config.data_params.val_examples_per_epoch or 50000
        )
        return train_dataset, val_dataset

    def create_model(self):
        model = LogisticRegression(
            self.num_features, self.train_dataset.NUM_CLASSES)
        return model.to(self.device)

    def configure_optimizers(self):
        if self.config.optim_params.supervised:
            parameters = list(self.model.parameters()) + list(self.encoder.parameters())
        else:
            parameters = self.model.parameters()
        if self.config.optim_params == 'adam':
            optim = torch.optim.Adam(parameters)
        else:
            optim = torch.optim.SGD(
                parameters,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def forward(self, inputs, train=True):
        batch_size = inputs.size(0)
        if train:
            inputs = self.viewmaker(inputs)
        if self.pretrain_config.model_params.resnet_small:
            embs = self.encoder(inputs, layer=6)
        else:
            embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch, train=True):
        _, inputs1, inputs2, label = batch
        logits = self.forward(inputs1, train=train)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch, train=True):
        _, inputs, inputs2, label = batch
        logits = self.forward(inputs, train=train)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = inputs.size(0)
        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=True)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(
                batch, train=True)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=False)
        num_correct, num_total = self.get_accuracies_for_batch(
            batch, train=False)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key]
                                         for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        progress_bar = {'acc': val_acc}
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc, 'progress_bar': progress_bar}

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)


class TransferExpertSystem(pl.LightningModule):
    '''Linear evaluation for SimCLR models'''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size

        self.encoder, self.system, self.pretrain_config = self.load_pretrained_model()
        resnet = self.pretrain_config.model_params.resnet_version
        if resnet == 'resnet18':
            if self.config.model_params.use_prepool:
                if self.pretrain_config.model_params.resnet_small:
                    num_features = 512 * 4 * 4
                else:
                    num_features = 512 * 2 * 2
            else:
                num_features = 512
        elif resnet == 'resnet50':
            num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')

        if not self.pretrain_config.model_params.resnet_small:
            raise NotImplementedError()

        if not self.config.optim_params.supervised:
            self.encoder = self.encoder.eval()
            frozen_params(self.encoder)

        self.num_features = num_features
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_model()

    def create_model(self):
        model = LogisticRegression(
            self.num_features, self.train_dataset.NUM_CLASSES)
        return model.to(self.device)

    def configure_optimizers(self):
        if self.config.optim_params.supervised:
            parameters = list(self.model.parameters()) + list(self.encoder.parameters())
        else:
            parameters = self.model.parameters()
        if self.config.optim_params == 'adam':
            optim = torch.optim.Adam(parameters)
        else:
            optim = torch.optim.SGD(
                parameters,
                lr=self.config.optim_params.learning_rate,
                momentum=self.config.optim_params.momentum,
                weight_decay=self.config.optim_params.weight_decay,
            )
        return [optim], []

    def get_losses_for_batch(self, batch, train=True):
        _, inputs1, inputs2, label = batch
        logits = self.forward(inputs1, train=train)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch, train=True):
        _, inputs, inputs2, label = batch
        logits = self.forward(inputs, train=train)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = inputs.size(0)
        return num_correct, num_total
        
    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=True)
        with torch.no_grad():
            num_correct, num_total = self.get_accuracies_for_batch(
                batch, train=True)
            metrics = {
                'train_loss': loss,
                'train_num_correct': num_correct,
                'train_num_total': num_total,
                'train_acc': num_correct / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=False)
        num_correct, num_total = self.get_accuracies_for_batch(
            batch, train=False)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct': num_correct,
            'val_num_total': num_total,
            'val_acc': num_correct / float(num_total)
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key]
                                         for elem in outputs]).float().mean()
        num_correct = sum([out['val_num_correct'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_acc = num_correct / float(num_total)
        metrics['val_acc'] = val_acc
        progress_bar = {'acc': val_acc}
        return {'val_loss': metrics['val_loss'], 'log': metrics, 'val_acc': val_acc, 'progress_bar': progress_bar}

    def create_datasets(self):
        print('Initializing train dataset.')
        if self.config.data_params.train_small:
            train_mode = 'train_small'
        else:
            train_mode = 'train'
        train_dataset = PAMAP2(
            mode=train_mode,
            examples_per_epoch=self.pretrain_config.data_params.train_examples_per_epoch or 10000,
            sensor_transforms=self.pretrain_config.data_params.sensor_transforms
        )
        print('Initializing validation dataset.')
        val_dataset = PAMAP2(
            mode='val',
            examples_per_epoch=self.pretrain_config.data_params.val_examples_per_epoch or 50000
        )
        return train_dataset, val_dataset
    
    def forward(self, inputs, train=True):
        batch_size = inputs.size(0)
        if self.pretrain_config.model_params.resnet_small:
            embs = self.encoder(inputs, layer=6)
        else:
            embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def load_pretrained_model(self):
        base_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = load_json(config_path)
        config = DotMap(config_json)

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(
            base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        if not self.config.optim_params.supervised:
            system.load_state_dict(checkpoint['state_dict'])

        encoder = system.model
        if not self.config.optim_params.supervised:
            encoder = system.model.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder, system, system.config

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)
