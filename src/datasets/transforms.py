import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(use_augmentation: bool, image_size: tuple = None):
    transform_list = []

    if use_augmentation:
        transform_list.extend([
            A.HueSaturationValue(always_apply=True),
            A.RandomBrightnessContrast(always_apply=True),
            A.HorizontalFlip(),
            A.RandomGamma(always_apply=True),
        ])

    if image_size:
        transform_list.append(
            A.Resize(height=image_size[0], width=image_size[1], always_apply=True)
        )

    transform_list.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    bbox_params = A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=0.0,
        min_visibility=0.0
    )

    return A.Compose(transform_list, bbox_params=bbox_params)