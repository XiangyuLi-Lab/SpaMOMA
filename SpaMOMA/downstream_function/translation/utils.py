
import scanpy as sc
import anndata as ad

import os
from scipy.io import mmread
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


from scipy.spatial import KDTree, cKDTree
from scipy.sparse import issparse, vstack, lil_matrix
import anndata as ad


def identify_overlapping_regions(adata_ref, adata_src, interp_X, distance_threshold=20):
    """
    Identify overlapping and non-overlapping regions between reference and source spatial data
    based on spatial distance threshold, and filter valid overlapping spots with non-zero expression.
    
    Parameters
    ----------
    adata_ref : anndata.AnnData
        Reference AnnData object with 'aligned_spatial' in obsm
    adata_src : anndata.AnnData
        Source AnnData object with 'aligned_spatial' in obsm
    interp_X : numpy.ndarray
        Interpolated expression matrix for reference spots
    distance_threshold : float, optional
        Maximum distance to define overlapping regions (default: 20)
    
    Returns
    -------
    ref_overlap_valid : numpy.ndarray (bool)
        Boolean mask for valid overlapping spots in reference (non-zero expression)
    ref_unique_mask : numpy.ndarray (bool)
        Boolean mask for non-overlapping spots in reference
    """
    
    ref_coords = adata_ref.obsm['aligned_spatial'].astype(np.float32)
    src_coords = adata_src.obsm['aligned_spatial'].astype(np.float32)
    
    src_tree = cKDTree(src_coords)
    distances, src_indices = src_tree.query(ref_coords, distance_upper_bound=distance_threshold)
    
    ref_overlap_mask = distances < distance_threshold
    unique_src_indices = np.unique(src_indices[ref_overlap_mask])
    ref_overlap_mask = np.isin(src_indices, unique_src_indices)
    ref_unique_mask = ~ref_overlap_mask  

    ref_overlap_valid = ref_overlap_mask & (interp_X.sum(axis=1) > 0)

    return ref_overlap_valid, ref_unique_mask

def build_spatial_relation_matrix(coords, ref_expr, mask_overlap, mask_non_overlap, k=10, similarity_metric="cosine"):
    """
    Build relation matrix based on expression similarity (instead of spatial distance) 
    between non-overlapping and overlapping regions of reference modality (e.g., RNA).
    
    Parameters
    ----------
    coords : numpy.ndarray
        Aligned spatial coordinates of all spots in reference slice (kept for compatibility, no actual calculation)
    ref_expr : numpy.ndarray or scipy.sparse matrix
        Expression matrix of reference slice (n_obs × n_genes), supports sparse/dense format
    mask_overlap : numpy.ndarray (bool)
        Boolean mask for overlapping regions
    mask_non_overlap : numpy.ndarray (bool)
        Boolean mask for non-overlapping regions
    k : int, optional
        Number of top similar overlapping spots to select for each non-overlapping spot (default: 10)
    similarity_metric : str, optional
        Similarity calculation method (options: 'cosine'/'pearson', default: 'cosine')
    
    Returns
    -------
    relation_matrix : numpy.ndarray
        Weighted relation matrix (n_non_overlap × n_overlap) where values represent 
        normalized similarity weights to overlapping spots
    
    Raises
    ------
    ValueError
        If unsupported similarity metric is provided
    """

    if issparse(ref_expr):
        ref_expr_dense = ref_expr.toarray().astype(np.float32)
    else:
        ref_expr_dense = ref_expr.astype(np.float32)

    ref_expr_dense = np.nan_to_num(ref_expr_dense, nan=0.0, posinf=0.0, neginf=0.0)

    ref_expr_norm = ref_expr_dense / (np.linalg.norm(ref_expr_dense, axis=1, keepdims=True) + 1e-8)
    expr_overlap = ref_expr_norm[mask_overlap]  
    expr_non_overlap = ref_expr_norm[mask_non_overlap]  
    n_non_overlap = len(expr_non_overlap)
    n_overlap = len(expr_overlap)

    if similarity_metric == "cosine":
        
        similarity_matrix = np.dot(expr_non_overlap, expr_overlap.T)  
    elif similarity_metric == "pearson":
        expr_overlap_centered = expr_overlap - expr_overlap.mean(axis=1, keepdims=True)
        expr_non_overlap_centered = expr_non_overlap - expr_non_overlap.mean(axis=1, keepdims=True)

        numerator = np.dot(expr_non_overlap_centered, expr_overlap_centered.T)
        denom_non_overlap = np.linalg.norm(expr_non_overlap_centered, axis=1, keepdims=True) + 1e-8
        denom_overlap = np.linalg.norm(expr_overlap_centered, axis=1, keepdims=True) + 1e-8
        similarity_matrix = numerator / (np.dot(denom_non_overlap, denom_overlap.T) + 1e-8)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}, options are 'cosine'/'pearson'")


    
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]  # (n_non_overlap × k)
    top_k_similarities = np.take_along_axis(similarity_matrix, top_k_indices, axis=1)  # (n_non_overlap × k)

    top_k_similarities = np.maximum(top_k_similarities, 0.0)
    top_k_weights = top_k_similarities / (np.sum(top_k_similarities, axis=1, keepdims=True) + 1e-8)

    relation_matrix = np.zeros((n_non_overlap, n_overlap), dtype=np.float32)
    for i in range(n_non_overlap):
        relation_matrix[i, top_k_indices[i]] = top_k_weights[i]

    return relation_matrix
