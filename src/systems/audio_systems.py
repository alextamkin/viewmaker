"""
Try some simple SimCLR inspired audio adaptations. Audio augmentations
include cropping, noise, pitch, and speed. We should fit this on Librispeech.
"""

import os
import math
import random
import librosa
import numpy as np
from dotmap import DotMap
from itertools import chain
from sklearn.metrics import f1_score
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from src.datasets.librispeech import LibriSpeech, LibriSpeechTwoViews, LibriSpeechTransfer
from src.datasets.voxceleb1 import VoxCeleb1
from src.datasets.audio_mnist import AudioMNIST
from src.datasets.google_speech import GoogleSpeechCommands
from src.datasets.fluent_speech import FluentSpeechCommands
from src.models.transfer import LogisticRegression
from src.models.resnet import resnet18
from src.models import resnet_small
from src.models.viewmaker import Viewmaker
from src.objectives.memory_bank import MemoryBank
from src.utils.utils import l2_normalize, frozen_params, free_params, load_json, compute_accuracy
from src.systems.image_systems import create_dataloader
from src.objectives.simclr import SimCLRObjective
from src.objectives.adversarial import AdversarialSimCLRLoss, AdversarialNCELoss
from src.objectives.infonce import NoiseConstrastiveEstimation

import pytorch_lightning as pl


class PretrainExpertInstDiscSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        # self.device = f'cuda:{config.gpu_device}' if config.cuda else 'cpu'
        self.train_dataset, self.val_dataset = self.create_datasets()
        self.model = self.create_encoder()
        self.memory_bank = MemoryBank(len(self.train_dataset), 
                                      self.config.model_params.out_dim) 
        self.train_ordered_labels = self.train_dataset.all_speaker_ids

    def create_datasets(self):
        print('Initializing train dataset.')
        train_dataset = LibriSpeech(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            small=self.config.data_params.small,
            input_size=self.config.data_params.input_size,
        )
        print('Initializing validation dataset.')
        val_dataset = LibriSpeech(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            test_url=self.config.data_params.test_url,
            input_size=self.config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def create_encoder(self):
        if self.config.model_params.resnet_small:
            encoder_model = resnet_small.ResNet18(
                self.config.model_params.out_dim,
                num_channels=1,
                input_size=64,
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
            mlp_dim = encoder_model.fc.weight.size(1)
            encoder_model.fc = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim),
                nn.ReLU(),
                encoder_model.fc,
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
        indices, inputs, _ = batch
        outputs = self.forward(inputs)
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
        neighbor_idxs = neighbor_idxs.cpu().numpy()
        
        neighbor_labels = self.train_ordered_labels[neighbor_idxs]
        neighbor_labels = torch.from_numpy(neighbor_labels).long()

        num_correct = torch.sum(neighbor_labels.cpu() == labels.cpu()).item()

        return num_correct, embs.size(0)
    
    def validation_step(self, batch, batch_idx):
        _, inputs, speaker_ids = batch
        outputs = self.model(inputs)
        num_correct, batch_size = self.get_nearest_neighbor_label(outputs, speaker_ids)
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
        return {'log': metrics, 'val_acc': val_acc} 

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)
    
    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)


class PretrainExpertSimCLRSystem(PretrainExpertInstDiscSystem):

    def create_datasets(self):
        train_dataset = LibriSpeechTwoViews(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            small=self.config.data_params.small,
            input_size=self.config.data_params.input_size,
        )
        val_dataset = LibriSpeech(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            test_url=self.config.data_params.test_url,
            input_size=self.config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def get_losses_for_batch(self, batch):
        indices, inputs1, inputs2, _ = batch
        outputs1 = self.forward(inputs1)
        outputs2 = self.forward(inputs2)
        loss_fn = SimCLRObjective(outputs1, outputs2, 
                                  t=self.config.loss_params.t)
        loss = loss_fn.get_loss()

        with torch.no_grad():  # for nearest neighbor
            new_data_memory = (l2_normalize(outputs1, dim=1) + 
                               l2_normalize(outputs2, dim=1)) / 2.
            self.memory_bank.update(indices, new_data_memory)

        return loss


class PretrainViewMakerInstDiscSystem(PretrainExpertInstDiscSystem):
    """
    InstDisc + Viewmaker
    """

    def __init__(self, config):
        super().__init__(config)
        self.view = self.create_viewmaker()

    def create_datasets(self):
        train_dataset = LibriSpeech(
            train=True, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            input_size=self.config.data_params.input_size,
        )
        val_dataset = LibriSpeech(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            test_url=self.config.data_params.test_url,
            input_size=self.config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def create_viewmaker(self):
        view_model = Viewmaker(
            num_channels=1,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            activation=self.config.model_params.generator_activation or 'relu',
            num_res_blocks=self.config.model_params.num_res_blocks,
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
        indices, inputs, _ = batch
        view = self.view(inputs)

        if self.config.model_params.view_clip:
            num_std = self.config.model_params.view_clip_num_std
            tot_std = num_std * self.train_dataset.normalize_stdev
            view_min = self.train_dataset.normalize_mean - tot_std
            view_max = self.train_dataset.normalize_mean + tot_std
            view = torch.clamp(view, view_min, view_max) 

        emb_dict = {
            'indices': indices,
            'view_embs': self.model(view),
        }
        return emb_dict

    def get_losses_for_batch(self, emb_dict):
        indices = emb_dict['indices']
        outputs = emb_dict['view_embs']
        loss_fn = AdversarialNCELoss(
            indices, outputs, self.memory_bank,
            k=self.config.loss_params.k,
            t=self.config.loss_params.t,
            m=self.config.loss_params.m,
            view_maker_loss_weight=self.config.loss_params.view_maker_loss_weight,
        )
        encoder_loss, view_maker_loss = loss_fn.get_loss()

        with torch.no_grad():
            new_data_memory = loss_fn.updated_new_data_memory()
            self.memory_bank.update(indices, new_data_memory)

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

            metrics = {
                'view_maker_loss': view_maker_loss,
                # 'view_bound_magnitude': self.view.bound_magnitude,
            }
            return {'loss': view_maker_loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        _, inputs, labels = batch
        outputs = self.model(inputs)
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


class PretrainViewMakerSimCLRSystem(PretrainExpertSimCLRSystem):
    """
    SimCLR + ViewMaker with Linf/L1 constraints.
    """

    def __init__(self, config):
        super().__init__(config)
        self.view = self.create_viewmaker()

    def create_datasets(self):
        train_dataset = LibriSpeechTwoViews(
            train=True, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            input_size=self.config.data_params.input_size,
        )
        val_dataset = LibriSpeech(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            small=self.config.data_params.small,
            test_url=self.config.data_params.test_url,
            input_size=self.config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def create_viewmaker(self):
        filter_size = self.train_dataset.FILTER_SIZE
        view_model = Viewmaker(
            num_channels=1,
            distortion_budget=self.config.model_params.view_bound_magnitude,
            activation=self.config.model_params.generator_activation or 'relu',
            num_res_blocks=self.config.model_params.num_res_blocks,
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
        indices, inputs, inputs2, _ = batch
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
        }
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

        return encoder_loss, view_maker_loss

    def get_view_bound_magnitude(self):
        if self.config.model_params.view_bound_linear_scale:
            batch_size = self.config.optim_params.batch_size 
            num_epochs = self.config.num_epochs
            num_steps = int(math.ceil(len(self.train_dataset) / batch_size)) * num_epochs
            view_bound_max = self.config.model_params.view_bound_max
            view_bound_min = self.config.model_params.view_bound_min
            iter_incr = (view_bound_max - view_bound_min) / num_steps
            return view_bound_min + self.global_step * iter_incr
        else:
            return self.config.model_params.view_bound_magnitude  # constant

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

            metrics = {
                'view_maker_loss': view_maker_loss,
                # 'view_bound_magnitude': self.view.bound_magnitude,
            }
            return {'loss': view_maker_loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        _, inputs, labels = batch
        outputs = self.model(inputs)
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


class BaseTransferExpertSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size

        self.encoder, self.pretrain_config = self.load_pretrained_model()
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
            if self.config.model_params.use_prepool:
                num_features = 2048 * 4 * 4
            else:
                num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')

        if not self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                cut_ix = -2
            else:
                cut_ix = -1
            # keep pooling layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:cut_ix])

        self.encoder = self.encoder.eval()
        frozen_params(self.encoder)

        self.train_dataset, self.val_dataset = self.create_datasets()
        self.num_features = num_features
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
        system.load_state_dict(checkpoint['state_dict'])

        encoder = system.model.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder, config

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)


class TransferExpertLibriSpeechSystem(BaseTransferExpertSystem):

    def create_datasets(self):
        train_dataset = LibriSpeechTransfer(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = LibriSpeechTransfer(
            train=False, 
            spectral_transforms=False, 
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def create_model(self):
        model = LogisticRegression(self.num_features, self.train_dataset.num_labels)
        return model.to(self.device)

    def configure_optimizers(self):
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

    def forward(self, inputs):
        batch_size = inputs.size(0)
        if self.pretrain_config.model_params.resnet_small:
            layer = 5 if self.config.model_params.use_prepool else 6
            embs = self.encoder(inputs, layer=layer)
            embs = F.avg_pool2d(embs, 2)
        else:
            embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch):
        _, inputs, label = batch
        logits = self.forward(inputs)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch):
        _, inputs, label = batch
        logits = self.forward(inputs)
        outputs = F.log_softmax(logits, dim=1)
        num_correct_top1, num_correct_top5 = compute_accuracy(outputs, label, topk=(1,5))
        num_total = inputs.size(0)
        return num_correct_top1, num_correct_top5, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        with torch.no_grad():
            num_correct_top1, num_correct_top5, num_total = self.get_accuracies_for_batch(batch)
            metrics = {
                'train_loss': loss,
                'train_num_correct_top1': num_correct_top1,
                'train_num_correct_top5': num_correct_top5,
                'train_num_total': num_total,
                'train_top1': num_correct_top1 / float(num_total),
                'train_top5': num_correct_top5 / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch)
        num_correct_top1, num_correct_top5, num_total = self.get_accuracies_for_batch(batch)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct_top1': num_correct_top1,
            'val_num_correct_top5': num_correct_top5,
            'val_num_total': num_total,
            'val_top1': num_correct_top1 / float(num_total),
            'val_top5': num_correct_top5 / float(num_total),
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct_top1 = sum([out['val_num_correct_top1'] for out in outputs])
        num_correct_top5 = sum([out['val_num_correct_top5'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_top1 = num_correct_top1 / float(num_total)
        val_top5 = num_correct_top5 / float(num_total)
        metrics['val_top1'] = val_top1
        metrics['val_top5'] = val_top5
        return {'val_loss': metrics['val_loss'], 'log': metrics, 
                'val_top1': val_top1,'val_top5': val_top5}


class TransferExpertVoxCeleb1System(TransferExpertLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = VoxCeleb1(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = VoxCeleb1(
            train=False, 
            spectral_transforms=False, 
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferExpertAudioMNISTSystem(TransferExpertLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = AudioMNIST(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = AudioMNIST(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferExpertGoogleSpeechCommandsSystem(TransferExpertLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = GoogleSpeechCommands(
            train=True, 
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = GoogleSpeechCommands(
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferExpertFluentSpeechCommandsSystem(TransferExpertLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = FluentSpeechCommands(
            self.config.data_params.caller_intent,
            train=True,
            spectral_transforms=self.config.data_params.spectral_transforms,
            wavform_transforms=not self.config.data_params.spectral_transforms,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = FluentSpeechCommands(
            self.config.data_params.caller_intent,
            train=False, 
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class BaseTransferViewMakerSystem(pl.LightningModule):

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
            if self.config.model_params.use_prepool:
                num_features = 2048 * 4 * 4
            else:
                num_features = 2048
        else:
            raise Exception(f'resnet {resnet} not supported.')

        if not self.pretrain_config.model_params.resnet_small:
            if self.config.model_params.use_prepool:
                cut_ix = -2
            else:
                cut_ix = -1
            self.encoder = nn.Sequential(*list(self.encoder.children())[:cut_ix])

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
        system.load_state_dict(checkpoint['state_dict'])
        
        encoder = system.model.eval()
        viewmaker = system.view.eval()

        for param in encoder.parameters():
            param.requires_grad = False

        for param in viewmaker.parameters():
            param.requires_grad = False

        return encoder, viewmaker, system, system.config

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, self.batch_size)

    def val_dataloader(self):
        return create_dataloader(self.val_dataset, self.config, self.batch_size, shuffle=False)


class TransferViewMakerLibriSpeechSystem(BaseTransferViewMakerSystem):

    def create_datasets(self):
        train_dataset = LibriSpeechTransfer(
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = LibriSpeechTransfer(
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset

    def create_model(self):
        model = LogisticRegression(self.num_features, self.train_dataset.num_labels)
        return model.to(self.device)

    def configure_optimizers(self):
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
            if self.pretrain_config.model_params.view_clip:
                num_std = self.pretrain_config.model_params.view_clip_num_std
                tot_std = num_std * self.train_dataset.normalize_stdev
                view_min = self.train_dataset.normalize_mean - tot_std
                view_max = self.train_dataset.normalize_mean + tot_std
                inputs = torch.clamp(inputs, view_min, view_max) 
        if self.pretrain_config.model_params.resnet_small:
            layer = 5 if self.config.model_params.use_prepool else 6
            embs = self.encoder(inputs, layer=layer)
            embs = F.avg_pool2d(embs, 2)
        else:
            embs = self.encoder(inputs)
        embs = embs.view(batch_size, -1)
        return self.model(embs)

    def get_losses_for_batch(self, batch, train=True):
        _, inputs, label = batch
        logits = self.forward(inputs, train=train)
        return F.cross_entropy(logits, label)

    def get_accuracies_for_batch(self, batch, train=True):
        _, inputs, label = batch
        logits = self.forward(inputs, train=train)
        outputs = F.log_softmax(logits, dim=1)
        num_correct_top1, num_correct_top5 = compute_accuracy(outputs, label, topk=(1,5))
        num_total = inputs.size(0)
        return num_correct_top1, num_correct_top5, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=True)
        with torch.no_grad():
            num_correct_top1, num_correct_top5, num_total = self.get_accuracies_for_batch(batch, train=True)
            metrics = {
                'train_loss': loss,
                'train_num_correct_top1': num_correct_top1,
                'train_num_correct_top5': num_correct_top5,
                'train_num_total': num_total,
                'train_top1': num_correct_top1 / float(num_total),
                'train_top5': num_correct_top5 / float(num_total),
            }
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, train=False)
        num_correct_top1, num_correct_top5, num_total = self.get_accuracies_for_batch(batch, train=False)
        return OrderedDict({
            'val_loss': loss,
            'val_num_correct_top1': num_correct_top1,
            'val_num_correct_top5': num_correct_top5,
            'val_num_total': num_total,
            'val_top1': num_correct_top1 / float(num_total),
            'val_top5': num_correct_top5 / float(num_total),
        })

    def validation_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            metrics[key] = torch.tensor([elem[key] for elem in outputs]).float().mean()
        num_correct_top1 = sum([out['val_num_correct_top1'] for out in outputs])
        num_correct_top5 = sum([out['val_num_correct_top5'] for out in outputs])
        num_total = sum([out['val_num_total'] for out in outputs])
        val_top1 = num_correct_top1 / float(num_total)
        val_top5 = num_correct_top5 / float(num_total)
        metrics['val_top1'] = val_top1
        metrics['val_top5'] = val_top5
        return {'val_loss': metrics['val_loss'], 'log': metrics,
                'val_top1': val_top1,'val_top5': val_top5}


class TransferViewMakerVoxCeleb1System(TransferViewMakerLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = VoxCeleb1(
            train=True, 
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = VoxCeleb1(
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferViewMakerAudioMNISTSystem(TransferViewMakerLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = AudioMNIST(
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = AudioMNIST(
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferViewMakerGoogleSpeechCommandsSystem(TransferViewMakerLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = GoogleSpeechCommands(
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = GoogleSpeechCommands(
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset


class TransferViewMakerFluentSpeechCommandsSystem(TransferViewMakerLibriSpeechSystem):

    def create_datasets(self):
        train_dataset = FluentSpeechCommands(
            self.config.data_params.caller_intent,
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        val_dataset = FluentSpeechCommands(
            self.config.data_params.caller_intent,
            train=False,
            spectral_transforms=False,
            wavform_transforms=False,
            input_size=self.pretrain_config.data_params.input_size,
        )
        return train_dataset, val_dataset
