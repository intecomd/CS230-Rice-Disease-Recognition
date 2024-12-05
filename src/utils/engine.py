import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils import tensorboard
from ignite import engine
from ignite import metrics
from ignite import handlers
from torch.utils.data import DataLoader
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events

# Define collate_fn at the top level
def collate_fn(batch):
    return tuple(zip(*batch))

def prepare_batch(batch, device=None, non_blocking=False):
    images, targets = batch
    images = [img.to(device=device, non_blocking=non_blocking) for img in images]
    targets = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
    return images, targets

def create_trainer(model: nn.Module, optimizer: optim.Optimizer, device=None, non_blocking: bool = False):
    if device:
        model.to(device)

    fn_prepare_batch = lambda batch: prepare_batch(batch, device=device, non_blocking=non_blocking)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        images, targets = fn_prepare_batch(batch)
        losses = model(images, targets)

        loss = sum(loss for loss in losses.values())

        loss.backward()
        optimizer.step()

        losses = {k: v.item() for k, v in losses.items()}
        losses['loss'] = loss.item()
        return losses

    return engine.Engine(_update)

def create_evaluator(model: nn.Module, metrics: Dict[str, metrics.Metric], device=None, non_blocking: bool = False):
    if device:
        model.to(device)

    fn_prepare_batch = lambda batch: prepare_batch(batch, device=device, non_blocking=non_blocking)

    def _inference(engine, batch):
        images, targets = fn_prepare_batch(batch)

        # Compute predictions in evaluation mode
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        # Compute losses in training mode
        model.train()
        with torch.no_grad():
            losses = model(images, targets)
            if isinstance(losses, dict):
                loss = sum(losses.values())
                losses = {k: v.item() for k, v in losses.items()}
                losses['loss'] = loss.item()
            else:
                losses = {'loss': losses.item()}

        # Return the model to evaluation mode
        model.eval()

        return outputs, targets, losses

    evaluator = engine.Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def attach_lr_scheduler(
    trainer: engine.Engine,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def update_lr(engine: engine.Engine):
        current_lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'epoch: {engine.state.epoch} - current lr: {current_lr}')
        writer.add_scalar('learning_rate', current_lr, engine.state.epoch)

        lr_scheduler.step()

def attach_training_logger(
    trainer: engine.Engine,
    writer: tensorboard.SummaryWriter,
    log_interval: int = 10,
):
    @trainer.on(engine.Events.ITERATION_COMPLETED)
    def log_training_loss(engine: engine.Engine):
        epoch_length = engine.state.epoch_length
        epoch = engine.state.epoch
        output = engine.state.output

        idx = engine.state.iteration
        idx_in_epoch = (engine.state.iteration - 1) % epoch_length + 1

        if idx_in_epoch % log_interval != 0:
            return

        msg = ''
        for name, value in output.items():
            msg += f'{name}: {value:.4f} '
            writer.add_scalar(f'training/{name}', value, idx)
        logging.info(f'epoch[{epoch}] - iteration[{idx_in_epoch}/{epoch_length}] ' + msg)

def attach_metric_logger(
    trainer: engine.Engine,
    evaluator: engine.Engine,
    data_name: str,
    data_loader: data.DataLoader,
    writer: tensorboard.SummaryWriter,
):
    @trainer.on(engine.Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        evaluator.run(data_loader)

        metrics = evaluator.state.metrics

        # Handle Average Losses separately
        if 'average_losses' in metrics:
            losses = metrics.pop('average_losses')
            for key, value in losses.items():
                writer.add_scalar(f'{data_name}/mean_{key}', value, engine.state.epoch)

        # Flatten and log other metrics
        def _flatten_metrics(metrics, parent_key='', sep='/'):
            items = []
            for k, v in metrics.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_metrics(v, new_key, sep=sep).items())
                elif isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        items.append((new_key, v.item()))
                    else:
                        for idx, val in enumerate(v):
                            items.append((f"{new_key}/class_{idx}", val.item()))
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_metrics = _flatten_metrics(metrics)
        message = ''
        for metric_name, metric_value in flat_metrics.items():
            writer.add_scalar(f'{data_name}/{metric_name}', metric_value, engine.state.epoch)
            message += f'{metric_name}: {metric_value:.3f} '

        logging.info(message)

def attach_model_checkpoint(trainer, model, optimizer, lr_scheduler, args):
    def global_step_transform(engine, event_name):
        return trainer.state.epoch

    handler = ModelCheckpoint(
        dirname='models',
        filename_prefix=args.model_tag,
        n_saved=5,
        create_dir=True,
        require_empty=False,
        global_step_transform=global_step_transform,
    )
    to_save = {
        'model': model.module if hasattr(model, 'module') else model,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
    }
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

def create_data_loaders(train_dataset, val_dataset, num_workers, batch_size):
    # Use the top-level collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader