import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import random
import shutil
import sys
import time
import warnings
from PIL import Image
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# sys.path.append(str(Path(__file__).parents[1]))

def visualize_matching_results(results):
    """
    Visualize results from run_matching function
    
    Args:
        results: Return dict from run_matching function
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(20, 15))
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(results["output_keypoints"])
    ax1.set_title("keypoints")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(results["output_matches_raw"])
    ax2.set_title(f"raw match (num: {results['matches']['num_raw_matches']})")
    ax2.axis('off')
    
    
    plt.tight_layout()
    plt.show()

    print("\nMatching statistics:")
    print(f"- Number of raw match points: {results['matches']['num_raw_matches']}")

    
    if results["geom_info"]:
        print("\nGeometric information:")
        if "Homography" in results["geom_info"]:
            print("- Homography matrix detected")
        if "Fundamental" in results["geom_info"]:
            print("- Fundamental matrix detected")





def visualize_filtered_matches(
    image0: np.ndarray,
    image1_aligned: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    point_size: int = 3,
    line_alpha: float = 0.6
):
    
    image0_rgb = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image1_rgb = cv2.cvtColor(image1_aligned, cv2.COLOR_BGR2RGB)
    
    h0, w0 = image0_rgb.shape[:2]
    h1, w1 = image1_rgb.shape[:2]
    max_h = max(h0, h1)
    total_w = w0 + w1
    

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, max_h)  
    ax.axis('off')
    
    ax.imshow(image0_rgb, extent=(0, w0, 0, h0))  
    ax.imshow(image1_rgb, extent=(w0, total_w, 0, h1))  
    
    kpts0_int = np.round(kpts0).astype(np.int32)
    kpts1_int = np.round(kpts1).astype(np.int32)
    
    kpts0_int[:, 1] = h0 - kpts0_int[:, 1]  
    kpts1_int[:, 1] = h1 - kpts1_int[:, 1]
    
    ax.scatter(
        kpts0_int[:, 0], kpts0_int[:, 1],
        s=point_size**2, c='red', alpha=1.0, edgecolors='none'
    )
    
    ax.scatter(
        kpts1_int[:, 0] + w0, kpts1_int[:, 1],
        s=point_size**2, c='blue', alpha=1.0, edgecolors='none'
    )
    
    num_matches = len(kpts0)
    if num_matches > 0:
        lines_x = np.empty(num_matches * 3)
        lines_y = np.empty(num_matches * 3)
        
        lines_x[::3] = kpts0_int[:, 0]
        lines_x[1::3] = kpts1_int[:, 0] + w0
        lines_x[2::3] = np.nan
        
        lines_y[::3] = kpts0_int[:, 1]
        lines_y[1::3] = kpts1_int[:, 1]
        lines_y[2::3] = np.nan
        
        ax.plot(
            lines_x, lines_y,
            color='cyan', linewidth=1, alpha=line_alpha,
            linestyle='-', solid_capstyle='round'
        )
    
    ax.set_title(
        f"Visualization of filtered valid matches\n" 
        f"Target image (left, red points) → Coarsely aligned source image (right, blue points)Number of matches: {num_matches}",
        fontsize=16, fontweight='bold', pad=20
    )
    
    plt.tight_layout(pad=0)
    plt.show()