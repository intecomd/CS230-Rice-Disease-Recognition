import logging

import torch
import torch.nn as nn
from torchvision.models import detection


class MaskRCNN(nn.Module):
    @staticmethod
    def load(state_dict):
        category_prefix = '_categories.'
        categories = [
            key[len(category_prefix):]
            for key in state_dict.keys() if key.startswith(category_prefix)
        ]

        model = MaskRCNN(categories)
        model.load_state_dict(state_dict)
        return model

    def __init__(self, categories):
        super().__init__()
        logging.info(f'Initializing MaskRCNN with categories: {categories}')

        self._categories = nn.ParameterDict({
            category: nn.Parameter(torch.empty(0)) for category in categories
        })
        num_classes = len(self._categories)

        self.model = detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        in_channels = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(
            in_channels, 256, num_classes
        )

    @property
    def categories(self):
        return list(self._categories.keys())

    def forward(self, images, targets=None):
        return self.model(images, targets)


def filter_by_threshold(result, bbox_thresh: float, mask_thresh: float):
    scores = result['scores'] > bbox_thresh
    filtered_result = {key: value[scores] for key, value in result.items()}
    filtered_result['masks'] = filtered_result['masks'][:, 0] >= mask_thresh
    return filtered_result