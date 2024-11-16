import numpy as np
from bm3d import bm3d_rgb


def cbm3d_denoise(image):
    # Normalize image to [0, 1]
    image_float = image.astype(np.float32) / 255.0
    # Apply BM3D denoising
    denoised_image = bm3d_rgb(image_float, sigma_psd=0.05)
    # Convert back to uint8
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)
    return denoised_image_uint8