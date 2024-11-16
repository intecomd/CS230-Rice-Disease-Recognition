import base64
import json
import logging
import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cbm3d_denoise import cbm3d_denoise
from .transforms import get_transforms


def decode_image_from_base64(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply CBM3D denoising
    image = cbm3d_denoise(image)
    return image


def generate_masks(shapes, width, height):
    for shape in shapes:
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array(shape['points']).reshape((-1, 1, 2))
        points = np.round(points).astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        yield mask


def generate_bounding_boxes(shapes):
    for shape in shapes:
        points = np.array(shape['points'])
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        yield [xmin, ymin, xmax, ymax]


class LabelMeDataset(Dataset):
    def __init__(self, directory: str, use_augmentation: bool):
        self.directory = pathlib.Path(directory)
        self.use_augmentation = use_augmentation
        assert self.directory.exists() and self.directory.is_dir()

        self.annotation_paths = []
        self.categories = set()

        for json_path in self.directory.rglob('*.json'):
            with open(json_path, 'r') as file:
                annotation = json.load(file)

            required_keys = [
                'version', 'flags', 'shapes', 'imagePath',
                'imageData', 'imageHeight', 'imageWidth'
            ]
            assert all(key in annotation for key in required_keys), (
                f"Missing keys in annotation file: {json_path}"
            )

            self.annotation_paths.append(json_path)
            for shape in annotation['shapes']:
                self.categories.add(shape['label'])

        self.categories = sorted(self.categories)
        logging.info(f'Loaded {len(self)} annotations from {self.directory}')
        logging.info(f'Use augmentation: {self.use_augmentation}')
        logging.info(f'Categories: {self.categories}')

        # Initialize transformations
        self.transforms = get_transforms(self.use_augmentation)

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, idx: int):
        json_path = self.annotation_paths[idx]
        logging.debug(f'Processing annotation: {json_path}')

        with open(json_path, 'r') as file:
            annotation = json.load(file)

        width = annotation['imageWidth']
        height = annotation['imageHeight']
        image = decode_image_from_base64(annotation['imageData'])
        assert image.shape == (height, width, 3)

        shapes = [
            shape for shape in annotation['shapes']
            if shape['shape_type'] == 'polygon' and len(shape['points']) > 2
        ]

        masks = list(generate_masks(shapes, width, height))
        bboxes = list(generate_bounding_boxes(shapes))
        labels = [self.categories.index(shape['label']) for shape in shapes]

        logging.debug('Applying transformations to image and targets')
        transformed = self.transforms(
            image=image,
            masks=masks,
            bboxes=bboxes,
            labels=labels
        )

        image = transformed['image']
        target = {
            'masks': torch.as_tensor(np.stack(transformed['masks']), dtype=torch.uint8),
            'labels': torch.as_tensor(transformed['labels'], dtype=torch.int64),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64),
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'boxes': torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
        }
        target['area'] = (
            (target['boxes'][:, 3] - target['boxes'][:, 1]) *
            (target['boxes'][:, 2] - target['boxes'][:, 0])
        )

        return image, target