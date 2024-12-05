# HARVEST: High-Accuracy Rice Verification and Evaluation System Technology


## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Environment Setup](#environment-setup)
4. [Usage](#usage)
    - [Training a New Model](#training-a-new-model)
    - [Inference](#inference)
    - [Dataset Inspection](#dataset-inspection)

---

## Overview

**HARVEST** (High-Accuracy Rice Verification and Evaluation System Technology) is an advanced system designed to accurately identify and evaluate rice diseases using state-of-the-art deep learning techniques. By integrating the CBM3D image denoising algorithm with a Mask R-CNN network enhanced by the Convolutional Block Attention Module (CBAM), HARVEST offers a robust solution for rice disease detection, crucial for maintaining global food security and agricultural sustainability.

Rice is a cornerstone of global food security and a major staple crop worldwide, holding significant economic and nutritional importance. Ensuring stable and increased rice yields is vital for global food security and agricultural sustainability. However, the frequent occurrence of rice diseases such as rice blast, bacterial leaf blight, and sheath blight poses severe threats to rice production, leading to substantial economic losses and undermining the health of the rice industry. Early and accurate identification of these diseases is critical for timely intervention and effective pest management, thereby safeguarding food security and promoting sustainable agricultural practices.

---

## Features

- **Custom Object Detection and Segmentation**: Fine-tune Mask R-CNN on annotated rice disease datasets.
- **CBM3D Denoising**: Integrates the CBM3D algorithm to preprocess images, effectively reducing noise while preserving essential details.
- **CBAM-Enhanced Backbone**: Incorporates the Convolutional Block Attention Module to focus on relevant features, improving detection accuracy in complex backgrounds.
- **Data Augmentation**: Implements a variety of augmentation techniques to enhance model robustness and generalization.
- **TensorBoard Integration**: Monitor training progress and metrics in real-time using TensorBoard.
- **Flexible Inference Options**: Visualize detection results directly or save them for further analysis.
- **Environment Management**: Utilizes Pyenv and `requirements.txt` for seamless environment setup and dependency management.

---

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:

- **Pyenv**: For managing Python versions. Install from [Pyenv GitHub](https://github.com/pyenv/pyenv).
- **Python 3.10.12**: Compatible with project dependencies.

### Installation

**Clone the Repository**

```bash
git clone https://github.com/intecomd/CS230-Rice-Disease-Recognition.git
cd CS230-Rice-Disease-Recognition
```

### Environment Setup

This project uses Pyenv and `requirements.txt` to manage its environment. Follow the steps below to set up the environment.

1. **Install Python 3.10.12 Using Pyenv**

    ```bash
    pyenv install 3.10.12
    pyenv local 3.10.12
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv harvest-env
    source harvest-env/bin/activate
    ```

3. **Upgrade Pip**

    ```bash
    pip install --upgrade pip
    ```

4. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Training a New Model

To train a new Mask R-CNN model on your custom rice disease dataset, follow these steps:

1. **Prepare Your Dataset**

    **Labeling**: Use [LabelMe](https://github.com/wkentaro/labelme) to annotate your images. Ensure you have a few hundred annotated images for optimal performance.

    ```bash
    labelme
    ```

2. **Inspect Your Dataset**

    Verify the annotations and augmentations applied using the `inspection.py` script.

    ```bash
    python inspection.py --dataset ~/datasets/my_rice_disease_val
    # To visualize with augmentations
    python inspection.py --dataset ~/datasets/my_rice_disease_val --use-augmentation
    ```

3. **Train the Model**

    Initiate training with the `train.py` script, specifying the training and validation data paths, model tag, and other hyperparameters.

    ```bash
    python train.py \
      --train ~/datasets/my_rice_disease_train \
      --val ~/datasets/my_rice_disease_val \
      --model-tag mask_rcnn_rice_disease \
      --num-epochs=30 \
      --num-workers=4 \
      --batch-size=4
    ```

    **Parameters:**

    - `--train`: Path to the training dataset.
    - `--val`: Path to the validation dataset.
    - `--model-tag`: Identifier for the trained model (used for saving).
    - `--num-epochs`: Number of training epochs.
    - `--num-workers`: Number of data loader workers.
    - `--batch-size`: Training batch size.
    - `--checkpoint`: *(Optional)* Path to a checkpoint file to resume training.
    - `--resume`: *(Optional)* Flag to resume training from a checkpoint.
    - `--debug`: *(Optional)* Enable debug mode for detailed logging.
    - `--initial-lr`: *(Optional)* Initial learning rate (default: 0.005).

    **Example:**

    ```bash
    python train.py \
      --train ~/datasets/rice_train \
      --val ~/datasets/rice_val \
      --model-tag harvest_mask_rcnn_rice_disease \
      --num-epochs=30 \
      --num-workers=4 \
      --batch-size=4 \
      --debug
    ```

4. **Monitor Training**

    Training logs and metrics are saved in the `logs` directory. You can monitor progress using TensorBoard:

    ```bash
    tensorboard --logdir=logs/
    ```

    Open the provided URL in your browser to visualize training metrics.

### Inference

Run the trained model on new images to perform object detection and instance segmentation.

1. **Display Output**

    ```bash
    python inference.py \
      --images ~/Pictures/my_rice_images \
      --model pretrained/model_mask_rcnn_rice_disease.pth \
      --display
    ```

2. **Save Output**

    ```bash
    python inference.py \
      --images ~/Pictures/my_rice_images \
      --model pretrained/model_mask_rcnn_rice_disease.pth \
      --save
    ```

    **Parameters:**

    - `--images`: Directory containing images for inference.
    - `--model`: Path to the trained model file.
    - `--display`: Flag to display output images.
    - `--save`: Flag to save output images with detections.
    - `--threshold`: Confidence threshold for detections (default: 0.5).
    - `--apply-denoise`: Apply CBM3D denoising to images before inference.

### Dataset Inspection

Visualize and verify dataset annotations using the `inspection.py` script.

```bash
python inspection.py \
  --dataset ~/datasets/my_rice_disease_val \
  --use-augmentation \
  --debug
```

**Parameters:**

- `--dataset`: Path to the dataset directory.
- `--use-augmentation`: Apply augmentations during inspection.
- `--debug`: Enable debug logging for detailed output.