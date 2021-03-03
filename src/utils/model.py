from typing import *
from pathlib import Path
from tqdm.notebook import tqdm
import os
import sys
import gc
import pickle
import random
import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch import optim
from torch.nn import Linear
import torch.nn.functional as F
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.extension import Extension, PRIORITY_READER
from pytorch_pfn_extras.training.manager import ExtensionsManager
from sklearn.metrics import average_precision_score
from logging import getLogger
from ignite.engine import Engine


# Metrics

def accuracy(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Computes multi-class classification accuracy"""
    assert y.shape[:-1] == t.shape, f"y {y.shape}, t {t.shape} is inconsistent."
    pred_label = torch.max(y.detach(), dim=-1)[1]
    count = t.nelement()
    correct = (pred_label == t).sum().float()
    acc = correct / count
    return acc


def accuracy_with_logits(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Computes multi-class classification accuracy"""
    assert y.shape == t.shape
    gt_label = torch.max(t.detach(), dim=-1)[1]
    return accuracy(y, gt_label)


def cross_entropy_with_logits(input, target, dim=-1):
    loss = torch.sum(- target * F.log_softmax(input, dim), dim)
    return loss.mean()


def cross_entropy(y: torch.Tensor, t: torch.Tensor, dim=-1) -> torch.Tensor:
    loss = torch.sum(- t * y, dim)
    return loss.mean()


def average_precision(y: torch.Tensor, t: torch.Tensor) -> float:
    """For binary classification"""
    assert y.shape == t.shape
    y = torch.softmax(y, dim=-1).detach().cpu().numpy()[:, 1]
    t = np.where(t.detach().cpu().numpy()[:, 1] >= 0.5, 1, 0)  # undo label smoothing
    return average_precision_score(t, y)  # probability of True labels


# Models

class CNNFixedPredictor(nn.Module):
    def __init__(self, cnn: nn.Module, num_classes: int = 2):
        super(CNNFixedPredictor, self).__init__()
        self.cnn = cnn
        self.lin = Linear(cnn.num_features, num_classes)
        print("cnn.num_features", cnn.num_features)

        # We do not learn CNN parameters.
        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.cnn(x)
        x = self.lin(x)
        return x


class Classifier(nn.Module):
    """two class classfication"""

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, x: torch.Tensor):
        outputs = self.predictor(x)
        # loss = self.lossfun(outputs, targets)
        # metrics = {
        #     f"{self.prefix}loss": loss.item(),
        #     f"{self.prefix}acc": accuracy_with_logits(outputs, targets).item(),
        #     f"{self.prefix}ap": average_precision(outputs, targets)
        # }
        # ppe.reporting.report(metrics, self)
        # return loss, metrics
        # outputs = torch.softmax(outputs, dim=-1)
        return outputs

    def predict(self, data_loader):
        pred = self.predict_proba(data_loader)
        label = torch.argmax(pred, dim=1)
        return label

    def predict_proba(self, data_loader):
        device: torch.device = next(self.parameters()).device
        y_list = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                if isinstance(batch, (tuple, list)):
                    # Assumes first argument is "image"
                    batch = batch[0].to(device)
                else:
                    batch = batch.to(device)
                y = self.predictor(batch)
                y = torch.softmax(y, dim=-1)
                y_list.append(y)
        pred = torch.cat(y_list)
        return pred


class EMA(object):
    """Exponential moving average of model parameters.
       From https://github.com/pfnet-research/kaggle-lyft-motion-prediction-4th-place-solution
    Ref
     - https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/optimizers/moving_average.py#L26-L103
     - https://anmoljoshi.com/Pytorch-Dicussions/

    Args:
        model (nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
        strict (bool): Apply strict check for `assign` & `resume`.
        use_dynamic_decay (bool): Dynamically change decay rate. If `True`, small decay rate is
            used at the beginning of training to move moving average faster.
    """  # NOQA

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        strict: bool = True,
        use_dynamic_decay: bool = True,
    ):
        self.decay = decay
        self.model = model
        self.strict = strict
        self.use_dynamic_decay = use_dynamic_decay
        self.logger = getLogger(__name__)
        self.n_step = 0

        self.shadow = {}
        self.original = {}

        # Flag to manage which parameter is assigned.
        # When `False`, original model's parameter is used.
        # When `True` (`assign` method is called), `shadow` parameter (ema param) is used.
        self._assigned = False

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def step(self):
        self.n_step += 1
        if self.use_dynamic_decay:
            _n_step = float(self.n_step)
            decay = min(self.decay, (1.0 + _n_step) / (10.0 + _n_step))
        else:
            decay = self.decay

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    # alias
    __call__ = step

    def assign(self):
        """Assign exponential moving average of parameter values to the respective parameters."""
        if self._assigned:
            if self.strict:
                raise ValueError("[ERROR] `assign` is called again before `resume`.")
            else:
                self.logger.warning(
                    "`assign` is called again before `resume`."
                    "shadow parameter is already assigned, skip."
                )
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
        self._assigned = True

    def resume(self):
        """Restore original parameters to a model.

        That is, put back the values that were in each parameter at the last call to `assign`.
        """
        if not self._assigned:
            if self.strict:
                raise ValueError("[ERROR] `resume` is called before `assign`.")
            else:
                self.logger.warning("`resume` is called before `assign`, skip.")
                return

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]
        self._assigned = False


class LRScheduler(Extension):
    """A thin wrapper to resume the lr_scheduler"""

    trigger = 1, 'iteration'
    priority = PRIORITY_READER
    name = None

    def __init__(self, optimizer: optim.Optimizer, scheduler_type: str, scheduler_kwargs: Mapping[str, Any]) -> None:
        super().__init__()
        self.scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_kwargs)

    def __call__(self, manager: ExtensionsManager) -> None:
        self.scheduler.step()

    def state_dict(self) -> None:
        return self.scheduler.state_dict()

    def load_state_dict(self, to_load) -> None:
        self.scheduler.load_state_dict(to_load)


def create_trainer(model, optimizer, device) -> Engine:
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        loss.backward()
        optimizer.step()
        return metrics
    trainer = Engine(update_fn)
    return trainer


def build_predictor(model_name: str, model_mode: str = "normal"):
    if model_mode == "normal":
        # normal configuration. train all parameters.
        return timm.create_model(model_name, pretrained=True, num_classes=2, in_chans=3)
    elif model_mode == "cnn_fixed":
        # normal configuration. train all parameters.
        # https://rwightman.github.io/pytorch-image-models/feature_extraction/
        timm_model = timm.create_model(model_name, pretrained=True, num_classes=0, in_chans=3)
        return CNNFixedPredictor(timm_model, num_classes=2)
    else:
        raise ValueError(f"[ERROR] Unexpected value model_mode={model_mode}")
