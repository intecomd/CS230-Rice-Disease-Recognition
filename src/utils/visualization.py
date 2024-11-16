import cv2
import numpy as np
import torch
from typing import Dict, List


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=1):
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    for i in range(0, length, gap * 2):
        start_point = (int(x1 + dx * i), int(y1 + dy * i))
        end_point = (int(x1 + dx * (i + gap)), int(y1 + dy * (i + gap)))
        cv2.line(img, start_point, end_point, color, thickness)


def draw_dotted_rectangle(img, pt1, pt2, color, thickness=1, gap=1):
    draw_dotted_line(img, pt1, (pt2[0], pt1[1]), color, thickness, gap)
    draw_dotted_line(img, (pt2[0], pt1[1]), pt2, color, thickness, gap)
    draw_dotted_line(img, pt2, (pt1[0], pt2[1]), color, thickness, gap)
    draw_dotted_line(img, (pt1[0], pt2[1]), pt1, color, thickness, gap)


def visualize_results(image: torch.Tensor, target: Dict[str, torch.Tensor], categories: List[str]):
    image_np = (image * 255).byte().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    red_color = (0, 0, 255)
    white_color = (255, 255, 255)

    for label, mask, bbox in zip(target['labels'], target['masks'], target['boxes']):
        label = label.item()
        mask = mask.cpu().numpy().astype(bool)
        category = categories[label]

        # Apply semi-transparent mask
        overlay = image_np.copy()
        overlay[mask] = red_color
        cv2.addWeighted(overlay, 0.3, image_np, 0.7, 0, image_np)

        # Draw bounding box
        x1, y1, x2, y2 = bbox.int().tolist()
        draw_dotted_rectangle(image_np, (x1, y1), (x2, y2), red_color, thickness=1, gap=1)

        # Draw label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        text_size, _ = cv2.getTextSize(category, font, font_scale, thickness)
        text_x = min(x1 + 2, image_np.shape[1] - text_size[0] - 1)
        text_y = min(y1 + text_size[1] + 2, image_np.shape[0] - 1)
        cv2.putText(image_np, category, (text_x, text_y), font, font_scale, white_color, thickness, cv2.LINE_AA)

    return image_np