import logging

from torch.utils.data import DataLoader, RandomSampler


def collate_fn(batch):
    return tuple(zip(*batch))


def create_data_loaders(train_dataset, val_dataset, num_workers, batch_size):
    logging.info(f'Creating data loaders with {num_workers} workers and batch size {batch_size}')
    data_loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **data_loader_params)

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=len(val_dataset))
    train_metrics_loader = DataLoader(train_dataset, sampler=train_sampler, **data_loader_params)

    val_loader = DataLoader(val_dataset, **data_loader_params)

    return train_loader, train_metrics_loader, val_loader