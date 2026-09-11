
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

from .utils import identify_overlapping_regions,build_spatial_relation_matrix

def translate(adata_target,adata_src,k_neighbors=5,overlap_distance_threshold=20):
    """
    Translate expression data from source to target spatial coordinates using neighbor interpolation 
    and spatial relation matrix for non-overlapping regions.
    
    Parameters
    ----------
    adata_target : anndata.AnnData
        Target AnnData object with spatial coordinates (aligned_spatial in obsm)
    adata_src : anndata.AnnData
        Source AnnData object with expression data and aligned spatial coordinates
    k_neighbors : int, optional
        Number of nearest neighbors for interpolation (default: 5)
        - Recommended: 5 for MBCS2, 4 for MT
    overlap_distance_threshold : float, optional
        Distance threshold to identify overlapping regions (default: 20)
        - Recommended: 20 for MBCS2, 90 for MT
    
    Returns
    -------
    anndata.AnnData
        Final translated AnnData object with expression values mapped to target coordinates
        - Contains translation metadata in uns['translation_info']
    """
    
    target_coords_df = pd.DataFrame(
        adata_target.obsm['aligned_spatial'],
        index=adata_target.obs.index
    )
    target_aligned_coords = target_coords_df.values

    src_coords_df = pd.DataFrame(
        adata_src.obsm['aligned_spatial'],
        index=adata_src.obs.index
    )
    src_aligned_coords = src_coords_df.values

    # print(f"Building KDTree for adata_src aligned coordinates ({src_aligned_coords.shape[0]} spots)...")
    kdtree = KDTree(src_aligned_coords)

    print(f"Translating omics-data from adata_src to adata_target...")
    distances, neighbor_indices = kdtree.query(target_aligned_coords, k=k_neighbors)
    src_X = adata_src.X
    is_sparse = issparse(src_X)
    n_target = adata_target.n_obs
    n_src = adata_src.n_obs

    # print(f"Batch computing average neighbor expression values for {n_target} obs ({'sparse matrix' if is_sparse else 'dense matrix'})...")

    if not is_sparse:
        neighbor_expr = src_X[neighbor_indices]
        interp_X = neighbor_expr.mean(axis=1)
        interp_X = np.asarray(interp_X)
    else:
        weights = lil_matrix((n_target, n_src), dtype=np.float32)
        for i in range(n_target):
            idx = neighbor_indices[i]
            weights[i, idx] = 1.0 / k_neighbors
        weights = weights.tocsr()
        interp_X = weights @ src_X
        interp_X = interp_X.toarray()

    adata_interp = ad.AnnData(
        X=interp_X,
        obs=adata_target.obs.copy(),
        var=adata_src.var.copy(),
        obsm=adata_target.obsm.copy(),
        uns=adata_target.uns.copy()
    )
    adata_interp.obs['mean_neighbor_distance'] = distances.mean(axis=1)
    adata_interp.uns['src_modal_info'] = {
        "source_adata": "adata_src",
        "k_neighbors": k_neighbors,
        "alignment_basis": "aligned_spatial",
        "computation_mode": "vectorized"
    }
    

    target_overlap_mask, target_unique_mask = identify_overlapping_regions(
        adata_target, adata_src, interp_X=interp_X,distance_threshold=overlap_distance_threshold
    )
    # print(f"Non-overlap ratio: {(np.sum(target_unique_mask)/(np.sum(target_overlap_mask)+np.sum(target_unique_mask)))*100}%")
    
    target_coords = adata_target.obsm['aligned_spatial'].astype(np.float32)
    target_expr = adata_target.X  

    # print("Building expression similarity relation matrix from non-overlapping to overlapping regions...")
    relation_matrix = build_spatial_relation_matrix(
        coords=target_coords,
        ref_expr=target_expr,  
        mask_overlap=target_overlap_mask,
        mask_non_overlap=target_unique_mask,
        k=10,  
        similarity_metric="pearson"  
    )
    

    interp_overlap = interp_X[target_overlap_mask]
    # print("Inferring non-overlapping region ATAC expression using relation matrix...")
    interp_non_overlap = relation_matrix @ interp_overlap

    final_translated_X = np.zeros_like(interp_X, dtype=np.float32)
    final_translated_X[target_overlap_mask] = interp_X[target_overlap_mask]
    final_translated_X[target_unique_mask] = interp_non_overlap
    final_translated_X = np.maximum(final_translated_X, 0)


    adata_final = ad.AnnData(
        X=final_translated_X,
        obs=adata_target.obs.copy(),
        var=adata_src.var.copy(),
        obsm=adata_target.obsm.copy(),
        uns=adata_target.uns.copy()
    )
    adata_final.uns['translation_info'] = {
        "strategy": "overlap=interp, non-overlap=spatial_relation_matrix",
        "overlap_spot_num": np.sum(target_overlap_mask),
        "non_overlap_spot_num": np.sum(target_unique_mask),
        "relation_matrix_params": {
            "k_neighbors": 10,
            "sigma": 50
        },
        "interp_k_neighbors": k_neighbors
    }

        
    return adata_final  
