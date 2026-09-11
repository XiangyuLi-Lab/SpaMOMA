import squidpy as sq
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from PIL import Image
import os
import tensorflow as tf

from matplotlib import patches
import cv2
tf.compat.v1.disable_eager_execution()

from sklearn.cluster import KMeans

import random
import os
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.interpolate import griddata

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def visualize_RGB(
    adata,
    output_dir='./',
    img_size=2048,
    rgb_key='STAGATE',
    normalize_method='percentile',  # 'percentile' or 'minmax'
    pct_low=1.0,
    pct_high=99.0,
    pad=10,
    invert_y=False,
    show=True,
    save_name='RGB.png',
    interpolation_method='linear',  # 'linear'/'cubic'/'nearest'
    gaussian_sigma=1.0,
    max_distance_ratio=1.2,  
    background_color=(255, 255, 255),  
    save_img=True
):
    """
    Interpolate discrete spots to continuous region, limit background filling by distance
    """
    
    print(f'generating spatial pattern image... gaussian_sigma:{gaussian_sigma}, interpolation_method:{interpolation_method}, max_distance_ratio"{max_distance_ratio}')
    
    if rgb_key not in adata.obsm:
        raise ValueError(f"adata.obsm must contain {rgb_key} (n x 3).")
    rgb_matrix = np.asarray(adata.obsm[rgb_key], dtype=np.float32)

    if 'spatia' in adata.obsm:
        coords = np.asarray(adata.obsm['spatia'], dtype=np.float64)
    elif 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    else:
        raise ValueError("adata.obsm must contain spatial coordinates under 'spatia' or 'spatial'.")

    coords = coords[:, :2].astype(np.float64)
    n_spots = coords.shape[0]
    if n_spots != rgb_matrix.shape[0]:
        raise ValueError(f"Number of spots mismatch: coords {n_spots} vs STAGATE {rgb_matrix.shape[0]}")

    vals = rgb_matrix.copy()
    norm_vals = np.zeros_like(vals, dtype=np.float32)
    for ch in range(3):
        c = vals[:, ch]
        if normalize_method == 'percentile':
            lo = np.percentile(c, pct_low)
            hi = np.percentile(c, pct_high)
            norm = (c - lo) / (hi - lo) if hi != lo else np.zeros_like(c)
        else:
            lo, hi = c.min(), c.max()
            norm = (c - lo) / (hi - lo) if hi != lo else np.zeros_like(c)
        norm_vals[:, ch] = np.clip(norm, 0.0, 1.0)

    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    scale = (img_size - 2 * pad) / max(span_x, span_y)

    out_w = int(np.round(span_x * scale)) + 2 * pad
    out_h = int(np.round(span_y * scale)) + 2 * pad

    grid_x, grid_y = np.meshgrid(
        np.linspace(pad, out_w - pad, out_w - 2 * pad),
        np.linspace(pad, out_h - pad, out_h - 2 * pad)
    )
    grid_points = np.column_stack((grid_x.flatten(), grid_y.flatten()))

    px = ((coords[:, 0] - min_x) * scale) + pad
    py = ((coords[:, 1] - min_y) * scale) + pad
    if invert_y:
        py = (out_h - 1) - py
    spot_points = np.column_stack((px, py))

    adata.obs['my_pixel_x'] = px
    adata.obs['my_pixel_y'] = py
    
    tree = KDTree(spot_points)
    avg_dist = np.mean(tree.query(spot_points, k=2)[0][:, 1]) 
    max_distance = avg_dist * max_distance_ratio 

    distances, _ = tree.query(grid_points)
    mask = distances <= max_distance  
    mask = mask.reshape(grid_x.shape)  

    rgb_channels = []
    for ch in range(3):
        ch_vals = griddata(
            points=spot_points,
            values=norm_vals[:, ch],
            xi=grid_points,
            method=interpolation_method,
            fill_value=0.0
        ).reshape(grid_x.shape)


        ch_vals = np.where(mask, ch_vals, 0)
        
        if gaussian_sigma > 0:
            ch_vals = gaussian_filter(ch_vals, sigma=gaussian_sigma)

        rgb_channels.append(np.clip(ch_vals, 0.0, 1.0))


    img = np.ones((out_h, out_w, 3), dtype=np.float32)
    img[pad:out_h-pad, pad:out_w-pad] = np.dstack(rgb_channels)

    img_arr = (img * 255).astype(np.uint8)
    bg_mask = np.all(img_arr[pad:out_h-pad, pad:out_w-pad] == 0, axis=2)
    for i in range(3):
        img_arr[pad:out_h-pad, pad:out_w-pad, i][bg_mask] = background_color[i]

    saved_path = os.path.join(output_dir, save_name)
    if save_img==True:
        print('saving image...')
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(img_arr).save(saved_path)

    if show:
        plt.figure(figsize=(3, 3 * out_h / out_w))
        plt.imshow(img_arr)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.show()

    return img_arr, saved_path



def image_to_color_blocks(
    img_path: str,
    n_clusters: int = 5,  
    gaussian_kernel: tuple = (15, 15),  
    sigmaX: float = 10.0,  
    show_result: bool = True
) -> np.ndarray:
    """
    Read image and convert to blurred + color-blocked effect
    Args:
        n_clusters: Number of color blocks (controls block size)
        gaussian_kernel: Gaussian blur kernel (odd numbers only, (h,w))
        sigmaX: Blur standard deviation (enhances blur effect)
        show_result: Whether to show comparison plot
    Returns:
        Color-blocked image (numpy array, RGB format)
    """

    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    h, w, c = img_rgb.shape


    img_blur = cv2.GaussianBlur(img_rgb, ksize=gaussian_kernel, sigmaX=sigmaX)

    img_flatten = img_blur.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(img_flatten)
    cluster_centers = kmeans.cluster_centers_.astype(np.uint8)  
    img_blocks = cluster_centers[cluster_labels].reshape(h, w, 3)

    img_blocks_smooth = cv2.GaussianBlur(img_blocks, ksize=(3, 3), sigmaX=1.0)

    if show_result:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img_blocks_smooth)
        plt.title(f"Color-Blocked Image ({n_clusters} blocks)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return img_blocks_smooth



