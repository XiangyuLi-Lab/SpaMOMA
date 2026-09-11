import matplotlib.pyplot as plt
import random
import numpy as np

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math

def affine_orig_to_resized(affine_orig, orig_size_A, orig_size_B, resized):
    """
    affine_orig: torch.Tensor or np.array shape (2,3) mapping A_orig -> B_orig (pixel coords)
    orig_size_A: (w0, h0) original A size
    orig_size_B: (w1, h1) original B size
    resized: (w_r, h_r) final resized dimension both images share
    """
    # ensure numpy
    if isinstance(affine_orig, torch.Tensor):
        affine_orig = affine_orig.cpu().numpy()
    a = np.eye(3, dtype=np.float32)
    a[:2, :] = affine_orig  # 3x3 homogeneous affine (last row [0,0,1])
    w0, h0 = orig_size_A
    w1, h1 = orig_size_B
    w_r, h_r = resized

    sx0 = w_r / float(w0)
    sy0 = h_r / float(h0)
    sx1 = w_r / float(w1)
    sy1 = h_r / float(h1)

    S_A = np.diag([sx0, sy0, 1.0]).astype(np.float32)
    S_B = np.diag([sx1, sy1, 1.0]).astype(np.float32)

    # affine_resized = S_B * a * inv(S_A)
    inv_SA = np.linalg.inv(S_A)
    a_res = S_B.dot(a).dot(inv_SA)
    return a_res[:2, :].astype(np.float32)


# ---- helper: apply 2x3 affine to pixel coords (x,y) (order: x horizontal, y vertical) ----
def apply_affine_to_points(affine2x3, pts):
    """
    affine2x3: (2,3)
    pts: (N,2) array (x,y) in pixel coords
    returns mapped_pts (N,2)
    """
    N = pts.shape[0]
    ones = np.ones((N,1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,3)
    mapped = pts_h.dot(affine2x3.T)  # (N,2)
    return mapped


def visualize_one_pair_from_dataset(dataset, idx=0, points=None, save_path='./affine_vis.png'):
    """
    dataset: your AffineWarpDataset instance
    idx: which pair to visualize
    points: optional list or array of points in resized A coords: [[x,y], ...]
            if None, we'll sample a small grid of points inside image
    """
    item = dataset[idx]
    # images returned by dataset are transformed Tensors (normalized). We need original resized PILs for plotting.
    # To get resized PIL for A we reload original image and resize; for B dataset already kept imgB in pairs but __getitem__ returns transformed tensors.
    # For robust approach: use dataset internal pairs array if available; else recreate from paths.
    # Here we assume your AffineWarpDataset has attribute .pairs storing (pA_path, imgB_pil, warp, affine_orig_tensor)
    try:
        pA_path, imgB_pil, warp_map_tensor, affine_tensor = dataset.pairs[idx]
    except Exception as e:
        raise RuntimeError("Dataset doesn't expose .pairs; adapt code to get original sizes and image B. Error: "+str(e))

    # load resized A PIL (as dataset.__getitem__ does)
    imgA_pil = Image.open(pA_path).convert('RGB')
    imgA_resized = imgA_pil.resize(dataset.resize, resample=Image.BILINEAR)
    imgB_resized = imgB_pil.resize(dataset.resize, resample=Image.BILINEAR)

    # sizes
    w0, h0 = Image.open(pA_path).convert('RGB').size  # original A size
    w1, h1 = imgB_pil.size  # original B size (before resizing)
    w_r, h_r = dataset.resize

    # get affine from dataset (it's in original pixel units)
    affine_orig = affine_tensor.cpu().numpy()  # shape (2,3)
    # compute affine mapped to resized pixel coords
    affine_res = affine_orig_to_resized(affine_orig, (w0,h0), (w1,h1), (w_r,h_r))

    # choose points in resized A coords
    if points is None:
        # sample a small grid inside image (avoid exact border rounding issues)
        gx = np.linspace(20, w_r-21, num=5)
        gy = np.linspace(20, h_r-21, num=4)
        pts = np.array([[x,y] for y in gy for x in gx], dtype=np.float32)  # (N,2)
    else:
        pts = np.array(points, dtype=np.float32)

    pts_mapped = apply_affine_to_points(affine_res, pts)

    # draw: show A with pts and B with mapped pts
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(imgA_resized)
    axes[0].set_title("Image A (resized) - chosen points")
    axes[0].scatter(pts[:,0], pts[:,1], c='r', s=40)
    for i,(x,y) in enumerate(pts):
        axes[0].text(x+3, y+3, str(i), color='yellow', fontsize=9, bbox=dict(facecolor='black', alpha=0.4, pad=1))

    axes[1].imshow(imgB_resized)
    axes[1].set_title("Image B (resized) - mapped points from A via affine")
    axes[1].scatter(pts_mapped[:,0], pts_mapped[:,1], c='r', s=40)
    for i,(x,y) in enumerate(pts_mapped):
        axes[1].text(x+3, y+3, str(i), color='yellow', fontsize=9, bbox=dict(facecolor='black', alpha=0.4, pad=1))

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.show()




