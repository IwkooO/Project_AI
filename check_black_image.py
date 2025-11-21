#!/usr/bin/env python3
"""Quick script to check if an image is actually black."""
from PIL import Image
import numpy as np
import sys

img_path = sys.argv[1] if len(sys.argv) > 1 else '/home/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/panorama/image_kDmg-De0yFP1Zwsal_dlzg.jpg'

img = Image.open(img_path)
arr = np.array(img)
print(f'Image: {img_path}')
print(f'Shape: {arr.shape}')
print(f'Pixel range: {arr.min()} to {arr.max()}')
print(f'Mean: {arr.mean():.2f}, Std: {arr.std():.2f}')

# Check the detection logic
if len(arr.shape) == 2:
    non_black = arr >= 10
else:
    non_black = np.any(arr[:, :, :3] >= 10, axis=2)

print(f'Non-black pixels (>=10): {np.sum(non_black):,} out of {non_black.size:,}')
print(f'Percentage non-black: {100 * np.sum(non_black) / non_black.size:.2f}%')
print(f'Is completely black: {not np.any(non_black)}')

