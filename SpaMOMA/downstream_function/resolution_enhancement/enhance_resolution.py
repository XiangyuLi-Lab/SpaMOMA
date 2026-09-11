

import pandas as pd
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from .magic import magic


def scale_contour(cnt, scale):
    """
    Scale a contour around its centroid.
    
    Parameters
    ----------
    cnt : numpy.ndarray
        OpenCV contour array (shape: [N, 1, 2])
    scale : float
        Scaling factor (1.0 = no scaling, >1 = enlarge, <1 = shrink)
    
    Returns
    -------
    numpy.ndarray
        Scaled contour array (same shape as input)
    """
    M = cv2.moments(cnt)
    if M['m00'] == 0: # Avoid division by zero if contour area is zero
        return cnt
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

def he_tissue_segmentation_with_internal_bg(
    img,
    blur_ksize=5,          
    a_channel_thresh=None,  
    min_area=10000,        
    min_hole_area=5000,     
    morph_ksize=15,        
    smooth_eps=0.001       
):
    """
    H&E image segmentation: Identify external background + internal background (holes/cavities).
    
    Parameters
    ----------
    img : numpy.ndarray
        Input H&E image (BGR format from OpenCV)
    blur_ksize : int, optional
        Kernel size for Gaussian blur (must be odd, default: 5)
    a_channel_thresh : int or None, optional
        Threshold for LAB A-channel segmentation. If None, Otsu's method is used (default: None)
    min_area : int, optional
        Minimum area for valid outer tissue contours (default: 10000)
    min_hole_area : int, optional
        Minimum area for valid internal hole contours (default: 5000)
    morph_ksize : int, optional
        Kernel size for morphological operations (default: 15)
    smooth_eps : float, optional
        Epsilon for contour approximation (smaller = more detailed, default: 0.001)
    
    Returns
    -------
    smooth_outer_cnt : numpy.ndarray
        Smoothed outer tissue contour (without internal holes)
    tissue_mask : numpy.ndarray
        Pure tissue mask (excludes external + internal background)
    smooth_hole_cnts : list of numpy.ndarray
        List of smoothed internal hole contours
    
    Raises
    ------
    RuntimeError
        If no tissue regions or valid outer contours are detected
    """
    
    img_blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    lab = cv2.cvtColor(img_blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)

    
    if a_channel_thresh is None:
        _, a_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, a_mask = cv2.threshold(a, a_channel_thresh, 255, cv2.THRESH_BINARY)
    _, s_mask = cv2.threshold(s, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(a_mask, s_mask)

    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize//2, morph_ksize//2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

   
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_CCOMP,       
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No tissue regions detected!")
    
    
    hierarchy = hierarchy[0]  

    
    outer_contours = []    
    hole_contours = []     
    
    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        cnt_area = cv2.contourArea(cnt)
        
        if h[3] == -1 and cnt_area >= min_area:
            outer_contours.append(cnt)
        elif h[3] != -1 and cnt_area >= min_hole_area:
            hole_contours.append(cnt)

    
    if not outer_contours:
        raise RuntimeError("No valid outer tissue contours detected!")
    outer_contour_areas = [cv2.contourArea(c) for c in outer_contours]
    max_outer_idx = np.argmax(outer_contour_areas)
    main_outer_cnt = outer_contours[max_outer_idx]

    outer_mask = np.zeros_like(mask)
    cv2.drawContours(outer_mask, [main_outer_cnt], -1, 255, -1)
    
    hole_mask = np.zeros_like(mask)
    cv2.drawContours(hole_mask, hole_contours, -1, 255, -1)
    
    tissue_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(hole_mask))

    outer_perimeter = cv2.arcLength(main_outer_cnt, closed=True)
    smooth_outer_cnt = cv2.approxPolyDP(
        main_outer_cnt, 
        epsilon=smooth_eps * outer_perimeter,
        closed=True
    )
    
    smooth_hole_cnts = []
    for hole_cnt in hole_contours:
        hole_perimeter = cv2.arcLength(hole_cnt, closed=True)
        smooth_hole = cv2.approxPolyDP(
            hole_cnt,
            epsilon=smooth_eps * hole_perimeter,
            closed=True
        )
        smooth_hole_cnts.append(smooth_hole)

    return smooth_outer_cnt, tissue_mask, smooth_hole_cnts



def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    if M['m00'] == 0:  
        return cnt
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled


def load_he_embedding_model(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()  
    return model

def extract_he_embedding_batch(
    x_pixel,
    y_pixel,
    image,
    beta,
    model,
    device="cuda",
    batch_size=64
):
    """
    Extract H&E image embeddings for a batch of spatial coordinates.
    
    Parameters
    ----------
    x_pixel : list of int
        List of x (row) pixel coordinates
    y_pixel : list of int
        List of y (column) pixel coordinates
    image : numpy.ndarray
        Input H&E image (BGR format)
    beta : int
        Patch size (beta x beta) for embedding extraction
    model : torch.nn.Module
        Pre-trained embedding model (ResNet50)
    device : str, optional
        Computation device ("cuda" or "cpu", default: "cuda")
    batch_size : int, optional
        Batch size for model inference (default: 64)
    
    Returns
    -------
    numpy.ndarray
        Normalized embeddings array (shape: [N, 512])
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    patches = []
    coords = list(zip(x_pixel, y_pixel))

    for x, y in coords:
        x, y = int(x), int(y)
        row1 = max(0, x - beta // 2)
        row2 = min(image.shape[0], x + beta // 2)
        col1 = max(0, y - beta // 2)
        col2 = min(image.shape[1], y + beta // 2)

        patch = image[row1:row2, col1:col2, :]  
        patch = cv2.resize(patch, (beta, beta))  
        patch = transform(patch)               
        patches.append(patch)

    patches = torch.stack(patches) 

    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, patches.shape[0], batch_size):
            batch = patches[i:i+batch_size].to(device)
            emb = model(batch).squeeze(-1).squeeze(-1)  # [B, 512]
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()

    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings


def imputation(img, raw, cnt, hole_cnts, genes, res=10,  k=4, num_nbs=4, patch_size=64):
    """
    Impute gene expression for pseudo spots in tissue regions.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input H&E image (BGR format)
    raw : anndata.AnnData
        Original spatial transcriptomics data (known spots)
    cnt : numpy.ndarray
        Outer tissue contour
    hole_cnts : list of numpy.ndarray
        List of internal hole contours
    genes : list of str
        List of gene names to impute
    res : int, optional
        Resolution (step size) for pseudo spot grid (default: 10)
    k : int or float, optional
        Weight decay factor for distance weighting (default: 4)
    num_nbs : int, optional
        Number of nearest neighbors for imputation (default: 4)
    
    Returns
    -------
    anndata.AnnData
        Imputed AnnData object with pseudo spots
    
    Raises
    ------
    RuntimeError
        If no valid pseudo spots or known spots in tissue region
    """
    
    tissue_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(tissue_mask, [cnt], -1, 255, thickness=-1)
    cv2.drawContours(tissue_mask, hole_cnts, -1, 0, thickness=-1)
    
    cnt_enlarged = scale_contour(cnt, 1.05)
    tissue_mask_enlarged = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(tissue_mask_enlarged, [cnt_enlarged], -1, 255, thickness=-1)
    cv2.drawContours(tissue_mask_enlarged, hole_cnts, -1, 0, thickness=-1)

    x_max, y_max = img.shape[0], img.shape[1]
    x_list = list(range(int(res), x_max, int(res)))
    y_list = list(range(int(res), y_max, int(res)))
    x = np.repeat(x_list, len(y_list)).tolist()
    y = y_list * len(x_list)
    sudo = pd.DataFrame({"x": x, "y": y})
    

    valid_idx = [
        i for i in sudo.index 
        if (0 <= sudo.x[i] < tissue_mask_enlarged.shape[0]) and 
           (0 <= sudo.y[i] < tissue_mask_enlarged.shape[1]) and 
           (tissue_mask_enlarged[sudo.x[i], sudo.y[i]] == 255)
    ]
    sudo = sudo.loc[valid_idx].reset_index(drop=True)
    print(f"Number of valid sudo spots in tissue region: {len(sudo)}")
    
    if len(sudo) == 0:
        raise RuntimeError("No valid known spots in tissue region! Cannot perform imputation")
    
    b = patch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = load_he_embedding_model().to(device)
    
    sudo_embeddings = extract_he_embedding_batch(
        x_pixel=sudo.x.tolist(),
        y_pixel=sudo.y.tolist(),
        image=img,
        beta=b,
        model=embedding_model,
        device=device,
        batch_size=64
    )
    
    sudo_embeddings = sudo_embeddings / np.linalg.norm(sudo_embeddings, axis=1, keepdims=True)
    sudo["embedding"] = list(sudo_embeddings)  

    
    known_adata = raw[:, raw.var.index.isin(genes)]
    known_adata.obs["x"] = known_adata.obs["pixel_col"]  
    known_adata.obs["y"] = known_adata.obs["pixel_row"]  
    
    known_valid_idx = [
        i for i in known_adata.obs.index
        if (0 <= known_adata.obs["x"][i] < tissue_mask.shape[0]) and 
           (0 <= known_adata.obs["y"][i] < tissue_mask.shape[1]) and 
           (tissue_mask[int(known_adata.obs["x"][i]), int(known_adata.obs["y"][i])] == 255)
    ]
    known_adata = known_adata[known_valid_idx].copy()
    print(f"Number of valid known spots in tissue region: {len(known_adata)}")
    
    if len(known_adata) == 0:
        raise RuntimeError("No valid known spots in tissue region! Cannot perform imputation")
    
    known_embeddings = extract_he_embedding_batch(
        x_pixel=known_adata.obs["x"].astype(int).tolist(), 
        y_pixel=known_adata.obs["y"].astype(int).tolist(), 
        image=img, beta=b, model=embedding_model, device=device, batch_size=64
    )
    known_embeddings = known_embeddings / np.linalg.norm(known_embeddings, axis=1, keepdims=True)
    known_adata.obs["embedding"] = list(known_embeddings)  

 
    dis = np.zeros((sudo.shape[0], known_adata.shape[0]))
    x_sudo, y_sudo = sudo["x"].values, sudo["y"].values
    x_known, y_known = known_adata.obs["x"].values, known_adata.obs["y"].values
    sudo_emb = np.array(sudo["embedding"].tolist())
    known_emb = np.array(known_adata.obs["embedding"].tolist())

    print("Total number of sudo points: ", sudo.shape[0])

    for i in range(sudo.shape[0]):
        if i % 1000 == 0:
            print("Calculating spot", i)
        
        spatial_dis = np.sqrt((x_sudo[i] - x_known)**2 + (y_sudo[i] - y_known)**2)
        
        cos_dis = 1 - np.dot(sudo_emb[i], known_emb.T)
        
        α = 0.6  
        dis[i] = α * (spatial_dis / np.max(spatial_dis)) + (1-α) * cos_dis  

    dis = pd.DataFrame(dis, index=sudo.index, columns=known_adata.obs.index)

    sudo_adata = sc.AnnData(np.zeros((sudo.shape[0], len(genes))))
    sudo_adata.obs = sudo
    sudo_adata.var = known_adata.var
    for i in range(sudo_adata.shape[0]):
        if i % 1000 == 0:
            print("Imputing spot", i)
        dis_tmp = dis.iloc[i, :].sort_values()
        nbs = dis_tmp[0:num_nbs]
        dis_tmp = (nbs.to_numpy() + 0.1) / np.min(nbs.to_numpy() + 0.1)
        if isinstance(k, int):
            weights = ((1/(dis_tmp**k)) / ((1/(dis_tmp**k)).sum()))
        else:
            weights = np.exp(-dis_tmp) / np.sum(np.exp(-dis_tmp))
        row_index = [known_adata.obs.index.get_loc(i) for i in nbs.index]
        sudo_adata.X[i, :] = np.dot(weights, known_adata.X[row_index, :])
    
    sudo_adata.obs["in_tissue"] = True
    sudo_adata.obsm["spatial"] = sudo[["y", "x"]].values 
    
    return sudo_adata



def enhance_resolution(counts,img_path,pseudo_spot_size=10,weight_decay=4,n_neighbors=4):

    magic_operator = magic.MAGIC()
    counts_denoise=counts.copy()
    counts_denoise.X = magic_operator.fit_transform(counts.X)
    counts=counts_denoise.copy()

    img = cv2.imread(img_path)
    resize_factor=1000/np.min(img.shape[0:2])
    resize_width=int(img.shape[1]*resize_factor)
    resize_height=int(img.shape[0]*resize_factor)
    counts.var_names_make_unique()
    counts.raw=counts
    if issparse(counts.X):
        counts.X=counts.X.todense()

    counts.obs['pixel_col']=counts.obsm['aligned_spatial'][:,1]
    counts.obs['pixel_row']=counts.obsm['aligned_spatial'][:,0]
    counts.obs['array_x']=counts.obs['array_row'].copy()
    counts.obs['array_y']=counts.obs['array_col'].copy()

    img_height = img.shape[0]  
    img_width = img.shape[1]  

    valid_mask = (
        (counts.obs['pixel_row'] >= 0) & 
        (counts.obs['pixel_row'] <= img_height) & 
        (counts.obs['pixel_col'] >= 0) & 
        (counts.obs['pixel_col'] <= img_width)
    )

    counts = counts[valid_mask].copy()
    
    cnt, tissue_mask, hole_cnts = he_tissue_segmentation_with_internal_bg(
        img,
        blur_ksize=5,
        a_channel_thresh=None,
        min_area=10000,
        min_hole_area=5000,  
        morph_ksize=25,
        smooth_eps=0.0005
    )


    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, [cnt], -1, (0, 255, 0), thickness=5)
    cv2.drawContours(img_with_contours, hole_cnts, -1, (0, 0, 255), thickness=3)

    resize_factor=1000/np.min(img.shape[0:2])
    resize_width=int(img.shape[1]*resize_factor)
    resize_height=int(img.shape[0]*resize_factor)
    img_with_contours = cv2.resize(img_with_contours, (resize_width, resize_height))

    enhanced_exp_adata = imputation(
        img=img, 
        raw=counts, 
        cnt=cnt, 
        hole_cnts=hole_cnts, 
        genes=counts.var.index.tolist(), 
        res=pseudo_spot_size, 
        k=weight_decay, 
        num_nbs=n_neighbors
    )

    print(f"Final enhanced data dimensions: {enhanced_exp_adata.shape}")
    enhanced_exp_adata.obsm['spatial']=enhanced_exp_adata.obs[['y','x']].values
    enhanced_exp_adata.obs['col']=enhanced_exp_adata.obs['x']
    enhanced_exp_adata.obs['row']=enhanced_exp_adata.obs['y']
    enhanced_exp_adata.obs_names=enhanced_exp_adata.obs['x'].astype(str)+'_'+enhanced_exp_adata.obs['y'].astype(str)
    embedding_matrix = np.vstack(enhanced_exp_adata.obs['embedding'].values)
    enhanced_exp_adata.obsm['HE_embedding'] = embedding_matrix
    del enhanced_exp_adata.obs['embedding']
    
    return enhanced_exp_adata
    



