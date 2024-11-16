import argparse
import logging
import pathlib

import cv2
import torch
from torchvision import transforms

from models import MaskRCNN, filter_by_threshold
from utils import visualize_results
from cbm3d_denoise import cbm3d_denoise


def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference script for Mask R-CNN.')
    parser.add_argument('--images', type=str, required=True, help='Directory containing images for inference.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections.')
    parser.add_argument('--save', action='store_true', help='Flag to save output images.')
    parser.add_argument('--display', action='store_true', help='Flag to display output images.')
    parser.add_argument('--apply-denoise', action='store_true', help='Apply CBM3D denoising to images.')
    return parser.parse_args()


def find_image_files(directory: pathlib.Path, extensions):
    assert directory.exists() and directory.is_dir(), f"Invalid directory: {directory}"
    for ext in extensions:
        yield from directory.rglob(f'*{ext}')


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Running inference on {device}')

    assert args.display or args.save, "At least one of --display or --save must be specified."

    # Load the trained model
    logging.info(f'Loading model from {args.model}')
    model_state = torch.load(args.model, map_location=device)
    model = MaskRCNN.load(model_state)
    model.to(device)
    model.eval()

    image_dir = pathlib.Path(args.images)
    image_extensions = ['.png', '.jpg', '.jpeg']

    # Function to filter results based on thresholds
    apply_thresholds = lambda res: filter_by_threshold(res, bbox_thresh=args.threshold, mask_thresh=args.threshold)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for image_path in find_image_files(image_dir, image_extensions):
        logging.info(f'Processing image: {image_path}')

        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f'Failed to read image: {image_path}')
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Optionally apply CBM3D denoising
        if args.apply_denoise:
            logging.info('Applying CBM3D denoising')
            image = cbm3d_denoise(image)

        image_tensor = transform(image).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            outputs = [apply_thresholds(output) for output in outputs]

        result_image = visualize_results(image_tensor[0], outputs[0], categories=model.categories)

        # Convert back to BGR for OpenCV display or saving
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        if args.save:
            output_path = pathlib.Path(f'results_{image_path.name}')
            logging.info(f'Saving output to {output_path}')
            cv2.imwrite(str(output_path), result_image)

        if args.display:
            cv2.imshow('Result', result_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                logging.info('Exiting...')
                break

    if args.display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()