from typing import Optional, Sequence, Tuple, Union

import torch
from pl_bolts.datamodules import STL10DataModule
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.metrics.functional import accuracy
from torch import device, nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from typing import Any


class SSLEvaluator(nn.Module):
    """Online evaluator that doesn't incorrectly apply output softmax layer."""

    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True),
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """

    def __init__(
        self,
        dataset: str,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer: Optimizer

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(
            pl_module.non_linear_evaluator.parameters(), lr=1e-4)

    def get_representations(self, pl_module: LightningModule, x: Tensor) -> Tensor:
        # Default augmentations already applied. Don't normalize or apply views.
        # Also, get representations from prepool layer.
        representations = pl_module(x, view=False)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch: Sequence, device: Union[str, device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch


        indices, x, img2, neg_img, y, = batch

        # last input is for online eval
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(
            representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        train_acc = accuracy(F.softmax(mlp_preds, dim=1), y)
        pl_module.log('online_train_acc', train_acc,
                      on_step=True, on_epoch=False)
        pl_module.log('online_train_loss', mlp_loss,
                      on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(
            representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        val_acc = accuracy(F.softmax(mlp_preds, dim=1), y)
        pl_module.log('online_val_acc', val_acc, on_step=False,
                      on_epoch=True, sync_dist=True, prog_bar=True)
        pl_module.log('online_val_loss', mlp_loss, on_step=False,
                      on_epoch=True, sync_dist=True)


class SSLOnlineEvaluatorSTL10(SSLOnlineEvaluator):
    '''Wrapper module for SSLOnlineEvaluator for STL10 datasets, which require online training
    and evaluation on labeled splits of STL10.
    Args:
        datamodule: initialized STL10DataModule, which contains functions that return labeled dataloader splits.
    '''

    def __init__(self, datamodule: STL10DataModule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.dataset == 'stl10', 'SSLOnlineEvaluatorSTL10 should only be called with dataset=stl10.'

        # Switch dataset so SSLOnlineEvaluator doesn't do this bs here:
        # https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/callbacks/ssl_online.py#L75
        self.dataset = ''

        # Initialise labeled dataloaders.
        self.train_dataloader = datamodule.train_dataloader_labeled()
        self.val_dataloader = datamodule.val_dataloader_labeled()

    # Nullify on_train_batch_end.
    def on_train_batch_end(self, *args, **kwargs):
        return

    # Nullify on_validation_batch_end.
    def on_validation_batch_end(self, *args, **kwargs):
        return

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any):
        '''Trains online MLP with labeled train_dataloader. Adapted from
        https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/callbacks/ssl_online.py#L88.'''
        train_loss = 0.0
        train_acc = 0.0
        for batch in self.train_dataloader:
            x, y = self.to_device(batch, pl_module.device)

            with torch.no_grad():
                representations = self.get_representations(pl_module, x)

            representations = representations.detach()

            # Forward pass.
            mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
            mlp_loss = F.cross_entropy(mlp_preds, y)

            # Update finetune weights.
            mlp_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log metrics.
            train_loss += mlp_loss.item()
            train_acc += accuracy(F.softmax(mlp_preds, dim=1), y).item()  # NOTE: pl_bolts applies softmax in evaluator

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        pl_module.log('online_train_loss', train_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_train_acc', train_acc, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        '''Evaluates online MLP with labeled val_dataloader. Adapted from
        https://github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/callbacks/ssl_online.py#L118.'''
        val_loss = 0.0
        val_acc = 0.0
        for batch in self.val_dataloader:
            x, y = self.to_device(batch, pl_module.device)

            with torch.no_grad():
                representations = self.get_representations(pl_module, x)

                # Forward pass.
                mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
                mlp_loss = F.cross_entropy(mlp_preds, y)

                # Log metrics.
                val_loss += mlp_loss.item()
                val_acc += accuracy(F.softmax(mlp_preds, dim=1), y).item()  # NOTE: pl_bolts applies softmax in evaluator

        val_loss /= len(self.val_dataloader)
        val_acc /= len(self.val_dataloader)
        pl_module.log('online_val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)