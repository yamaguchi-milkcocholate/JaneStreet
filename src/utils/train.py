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
import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from ignite.engine import Events, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.metrics import AveragePrecision


def score_function(engine):
    """
    Due to maximizing score_function, this returns (-1) x loss
    """
    val_loss = engine.state.metrics['BCE']
    return -val_loss


def discreted_output_transform(output):
    y_pred, y = output
    y_pred = torch.argmax(y_pred, dim=-1)
    y = torch.argmax(y, dim=-1)
    return y_pred, y


def probability_output_transform(output):
    y_pred, y = output
    y_pred = torch.softmax(y_pred, dim=1)[:, 1]
    y = torch.argmax(y, dim=1)
    return y_pred, y


def train(epochs: int, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, criterion: Callable,
          device: str, lr: float, patience: int, lr_decay: float, lr_scheduler: str, lr_scheduler_kwargs: Dict[str, Any]):
    
    model.to(torch.device(device))
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    
    trainer = create_supervised_trainer(
        model, 
        optimizer, 
        criterion, 
        device=device
    )
    
    scheduler = LRScheduler(getattr(optim.lr_scheduler, lr_scheduler)(optimizer, **lr_scheduler_kwargs))
    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    
    pbar = ProgressBar(False)
    pbar.attach(trainer)
    
    train_evaluator = create_supervised_evaluator(
        model,
        metrics={'ACC': Accuracy(discreted_output_transform), 'BCE': Loss(criterion), 'AP': AveragePrecision(probability_output_transform)},
        device=device
    )
    valid_evaluator = create_supervised_evaluator(
        model,
        metrics={'ACC': Accuracy(discreted_output_transform), 'BCE': Loss(criterion), 'AP': AveragePrecision(probability_output_transform)},
        device=device
    )
    
    history = {col: list() for col in ['epoch', 'elapsed time', 'iterations', 'lr', 'train BCE', 'valid BCE', 'train ACC', 'valid ACC', 'train AP', 'valid AP']}

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        
        history['train BCE'] += [train_evaluator.state.metrics['BCE']]
        history['train ACC'] += [train_evaluator.state.metrics['ACC']]
        history['train AP'] += [train_evaluator.state.metrics['AP']]
        

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        valid_evaluator.run(valid_loader)
        
        history['epoch'] += [valid_evaluator.state.epoch]
        history['iterations'] += [valid_evaluator.state.epoch_length]
        history['elapsed time'] += [0 if len(history['elapsed time']) == 0 else history['elapsed time'][-1] + valid_evaluator.state.times['COMPLETED']]
        history['lr'] += [scheduler.get_param()]
        
        history['valid BCE'] += [valid_evaluator.state.metrics['BCE']]
        history['valid ACC'] += [valid_evaluator.state.metrics['ACC']]
        history['valid AP'] += [valid_evaluator.state.metrics['AP']]
        

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_progress_bar(engine):
        pbar.log_message(
            f"train BCE: {history['train BCE'][-1]:.2f} " \
            + f"train ACC: {history['train ACC'][-1]:.2f} " \
            + f"train AP: {history['train AP'][-1]:.2f} " \
            + f"valid BCE: {history['valid BCE'][-1]:.2f} " \
            + f"valid ACC: {history['valid ACC'][-1]:.2f} " \
            + f"valid AP: {history['valid AP'][-1]:.2f}"
        )
    

    # Early stopping
    handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    return pd.DataFrame(history)