import argparse
import logging
import torch
from torch import nn, optim
from torch.utils import tensorboard
from datasets import LabelMeDataset
from utils import (
    create_trainer, create_evaluator, attach_lr_scheduler,
    attach_training_logger, attach_metric_logger, attach_model_checkpoint, create_data_loaders
)
from utils import AverageLoss, MeanAveragePrecision, MeanIoU
from models import get_instance_segmentation_model, MaskRCNN

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Mask-RCNN for Rice Disease Identification")
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--val', type=str, required=True, help='Path to validation data')
    parser.add_argument('--model-tag', type=str, required=True, help='Model tag for saving')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file to resume training')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--initial-lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    return parser.parse_args()

def main():
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler(f"{args.model_tag}_training.log"),
            logging.StreamHandler()
        ]
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Running training on {device}')

    # Prepare datasets and loaders
    train_dataset = LabelMeDataset(args.train, use_augmentation=True)
    val_dataset = LabelMeDataset(args.val, use_augmentation=False)
    assert train_dataset.categories == val_dataset.categories, "Training and validation categories do not match."

    train_loader, train_metrics_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, args.num_workers, args.batch_size
    )
    logging.info(f'Creating data loaders with {args.num_workers} workers and batch size {args.batch_size}')

    for batch in train_loader:
        images, targets = batch
        logging.info(f"Train Batch - Images: {len(images)}, Targets: {len(targets)}")
        break

    for batch in val_loader:
        images, targets = batch
        logging.info(f"Validation Batch - Images: {len(images)}, Targets: {len(targets)}")
        break

    num_classes = len(train_dataset.categories)

    # Initialize MaskRCNN model
    # model = MaskRCNN(train_dataset.categories)

    # Initialize MaskRCNN-CBAM model
    model = get_instance_segmentation_model(num_classes)

    model = nn.DataParallel(model).to(device)

    # Adjust the initial learning rate for SGD (Adjusted based on batch size)
    args.initial_lr = 0.005  

    # Initialize optimizer and scheduler with SGD and momentum
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.initial_lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    # Use ExponentialLR scheduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    start_epoch = 1

    if args.checkpoint and args.resume:
        logging.info(f'Loading checkpoint from {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device)

        model_state_dict = checkpoint

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        # Re-initialize optimizer and scheduler
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.initial_lr,
            momentum=0.9,
            weight_decay=0.0005,
        )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # Set the starting epoch or set manually if known
        start_epoch = 1
        logging.info(f'Resumed training from epoch {start_epoch}')

    trainer = create_trainer(model, optimizer, device=device)

    evaluator = create_evaluator(
        model, metrics={
            'average_losses': AverageLoss(device=device),
            'mean_precision': MeanAveragePrecision(device=device),
            'mean_iou': MeanIoU(class_metrics=True, device=device)
        }, device=device, non_blocking=True
    )

    writer = tensorboard.SummaryWriter(log_dir=f'logs/{args.model_tag}')

    # Attach event handlers
    attach_lr_scheduler(trainer, lr_scheduler, writer)
    attach_training_logger(trainer, writer)
    attach_metric_logger(trainer, evaluator, 'train', train_metrics_loader, writer)
    attach_metric_logger(trainer, evaluator, 'val', val_loader, writer)
    attach_model_checkpoint(trainer, model, optimizer, lr_scheduler, args)

    logging.info('Starting training...')

    # Calculate total epochs to run
    total_epochs = start_epoch + args.num_epochs - 1

    trainer.run(train_loader, max_epochs=total_epochs)

    # Save final checkpoint
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, f'{args.model_tag}_final.pth')

    writer.close()

    logging.info("Training and Evaluation Completed Successfully!")

if __name__ == '__main__':
    main()