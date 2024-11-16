from collections import defaultdict

from ignite.metrics import Metric


class AverageLoss(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super().__init__(output_transform=output_transform, device=device)
        self.reset()
    
    def reset(self):
        self.total_loss = defaultdict(float)
        self.sample_count = 0
    
    def update(self, output):
        loss_dict, batch_size = output
        self.sample_count += batch_size
        for key, value in loss_dict.items():
            self.total_loss[key] += value * batch_size
    
    def compute(self):
        if self.sample_count == 0:
            raise ValueError("AverageLoss must have at least one example before it can be computed.")
        average_loss = {key: value / self.sample_count for key, value in self.total_loss.items()}
        return average_loss