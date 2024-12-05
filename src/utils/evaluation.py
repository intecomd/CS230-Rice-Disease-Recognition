import torch
import torchmetrics
from ignite.metrics import Metric
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics import Recall
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import JaccardIndex as TorchMetricsJaccardIndex
from collections import defaultdict

class AverageLoss(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        super(AverageLoss, self).__init__(output_transform=output_transform, device=device)
        self.reset()

    def reset(self):
        self.total_loss = defaultdict(float)
        self.sample_count = 0

    def update(self, output):
        _, _, losses = output
        batch_size = 1  # Since losses are averaged over the batch
        self.sample_count += batch_size
        for key, value in losses.items():
            self.total_loss[key] += value * batch_size

    def compute(self):
        if self.sample_count == 0:
            raise ValueError("AverageLoss must have at least one example before it can be computed.")
        average_loss = {key: value / self.sample_count for key, value in self.total_loss.items()}
        return average_loss
    
class MeanAveragePrecision(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self.map_metric = torchmetrics.detection.MeanAveragePrecision()
        self.device = device
        super(MeanAveragePrecision, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.map_metric.reset()

    def update(self, output):
        outputs, targets, _ = output
        y_pred = [{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in pred.items()} for pred in outputs]
        y_true = [{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
        self.map_metric.update(y_pred, y_true)

    def compute(self):
        return self.map_metric.compute()
    
class MeanIoU(Metric):
    def __init__(self, class_metrics=False, output_transform=lambda x: x, device=None):
        self.device = device
        self.class_metrics = class_metrics
        self.iou_metric = IntersectionOverUnion(class_metrics=class_metrics).to(device)
        super(MeanIoU, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self.iou_metric.reset()

    def update(self, output):
        outputs, targets, _ = output

        preds = []
        gts = []

        for pred, target in zip(outputs, targets):
            pred_dict = {
                'boxes': pred['boxes'].to(self.device),
                'scores': pred['scores'].to(self.device),
                'labels': pred['labels'].to(self.device),
            }

            target_dict = {
                'boxes': target['boxes'].to(self.device),
                'labels': target['labels'].to(self.device),
            }

            # Add masks if available
            if 'masks' in pred:
                pred_dict['masks'] = pred['masks'].to(self.device)
            if 'masks' in target:
                target_dict['masks'] = target['masks'].to(self.device)

            preds.append(pred_dict)
            gts.append(target_dict)

        # Update the metric with the current batch
        self.iou_metric.update(preds, gts)

    def compute(self):
        return self.iou_metric.compute()