import argparse
import logging

import cv2

from datasets import LabelMeDataset
from utils import visualize_results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dataset inspection script.')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--use-augmentation', action='store_true', help='Apply augmentations during inspection.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    return parser.parse_args()


def log_tensor_stats(name, tensor):
    dtype = tensor.dtype
    stats = {
        'min': tensor.min().item(),
        'mean': tensor.float().mean().item(),
        'max': tensor.max().item(),
        'shape': tuple(tensor.shape),
        'dtype': dtype,
    }
    logging.info(f"{name} stats - shape: {stats['shape']}, dtype: {stats['dtype']}, min: {stats['min']:.4f}, mean: {stats['mean']:.4f}, max: {stats['max']:.4f}")


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    dataset = LabelMeDataset(args.dataset, args.use_augmentation)

    num_samples = len(dataset)
    for idx in range(num_samples):
        logging.info(f'Displaying sample {idx + 1}/{num_samples}')
        image, target = dataset[idx]

        for key, value in target.items():
            log_tensor_stats(key, value)

        result_image = visualize_results(image, target, categories=dataset.categories)

        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Result', result_image)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            logging.info('Exiting inspection.')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()