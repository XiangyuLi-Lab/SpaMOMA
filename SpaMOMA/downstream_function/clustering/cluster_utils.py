import scanpy as sc
import numpy as np
from scipy.spatial import cKDTree

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def run_leiden(adata, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata.obs[key_added] = adata.obs["leiden"]
            
            return adata
        
        this_step += 1
    
    print('Cannot find the number of clusters')
    adata.obs[key_added] = adata.obs["leiden"]
    return adata

def run_louvain(adata, n_cluster, use_rep="embeddings", key_added="louvain", range_min=0, range_max=3, max_steps=30, tolerance=0):
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata.obs[key_added] = adata.obs["louvain"]
            
            return adata
        
        this_step += 1
    
    print('Cannot find the number of clusters')
    adata.obs[key_added] = adata.obs["louvain"]
    return adata




 

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def run_mclust(adata, n_clusters=7, key='emb', add_key='mclust', use_pca=False, n_comps=20):
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
  
    if use_pca: 
        adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
    else:
        adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
    adata.obs[add_key] = adata.obs['mclust']
    
       
       
import numpy as np
from .sc3s._legacy import convert_dict_into_contigency_matrix, cluster_matrix_kmeans
def consensus_vote_from_existing(
    adata,
    cluster_cols,  
    n_clusters,  
    add_key=None
):

    trials_dict = {}
    n_cells = adata.n_obs
    for col in cluster_cols:
        labels = adata.obs[col].values
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype('category').cat.codes  

        trials_dict[col] = {'labels': labels}
    
    consensus_matrix = convert_dict_into_contigency_matrix(
        dict_object=trials_dict,
        true_n_clusters=0,  
        true_n_cells=n_cells
    )
    print('got consensus_matrix')

    consensus_labels = cluster_matrix_kmeans(
        consensus_matrix=consensus_matrix,
        n_clusters=n_clusters,
        random_state=42
    )
    print('got consensus_labels')
    
    adata.obs[add_key] = consensus_labels
    adata.obs[add_key] = adata.obs[add_key].astype('category')
    
    return adata



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import scipy.sparse as sp

def spatial_neighborhood_voting(
    adata,
    cluster_col="LeidenClusters",  
    spatial_key="aligned_spatial",
    k=5,                           
    outlier_threshold=0.8,         
    new_cluster_col="LeidenClusters_smoothed" 
):

    """
    Spatial neighborhood label voting smoothing: Correct only isolated discrete points while preserving continuous fine structures.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Scanpy AnnData object containing spatial transcriptomics data
    cluster_col : str, optional
        Column name of original cluster labels in adata.obs (default: "LeidenClusters")
    spatial_key : str, optional
        Key in adata.obsm for spatial coordinates (default: "aligned_spatial")
    k : int, optional
        Number of nearest neighbors. Smaller k preserves more details, larger k increases smoothing (recommended: 3-5)
    outlier_threshold : float, optional
        Threshold for outlier detection. Higher values are more conservative (only correct extremely discrete points) (default: 0.8)
    new_cluster_col : str, optional
        Column name to store smoothed cluster labels (default: "LeidenClusters_smoothed")
    
    Returns
    -------
    anndata.AnnData
        AnnData object with smoothed cluster labels added to obs
    """

    spatial_coords = adata.obsm[spatial_key].copy()
    cluster_labels = adata.obs[cluster_col].astype(str).values  
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)
    # Remove self (0th neighbor is the cell itself)
    indices = indices[:, 1:]  # Shape: (n_cells, k)
    

    smoothed_labels = cluster_labels.copy()
    for i in range(len(spatial_coords)):
        neighbor_labels = cluster_labels[indices[i]]
        label_counts = pd.Series(neighbor_labels).value_counts(normalize=True)
        top_label = label_counts.index[0]
        top_label_ratio = label_counts.iloc[0]
        
        if (cluster_labels[i] != top_label) and (top_label_ratio > outlier_threshold):
            smoothed_labels[i] = top_label
    
    adata.obs[new_cluster_col] = smoothed_labels
    adata.obs[new_cluster_col] = adata.obs[new_cluster_col].astype(cluster_labels.dtype)
    
    n_changed = (adata.obs[cluster_col] != adata.obs[new_cluster_col]).sum()
    # print(f"Corrected {n_changed} discrete points (total cells: {len(adata)})")
    
    return adata

def domain_label_transfer(
    batch_id,               
    full_adata,             # overlap+non-overlap region (full region)
    joint_cluster_adata,    # overlap region with PRESENT cluster (multiple batches)
    domain_key='LeidenClusters',
    seed=0
):
    """
    Transfer domain labels from overlapping regions to non-overlapping regions using machine learning.
    
    Parameters
    ----------
    batch_id : str/int
        Identifier of the current batch to process
    full_adata : anndata.AnnData
        Full dataset containing both overlapping and non-overlapping regions
    joint_cluster_adata : anndata.AnnData
        Overlapping region data with pre-computed cluster labels (multiple batches)
    domain_key : str, optional
        Column name for cluster/domain labels (default: "LeidenClusters")
    seed : int, optional
        Random seed for reproducibility (default: 0)
    
    Returns
    -------
    anndata.AnnData
        Full AnnData object with predicted labels for non-overlapping regions
    
    Raises
    ------
    ValueError
        If no valid training samples (overlap=True and with labels) exist for the batch
    """

    print(f"\n========== Processing batch: {batch_id} ==========")
    joint_batch_adata = joint_cluster_adata[joint_cluster_adata.obs['batch'] == batch_id].copy()
    full_adata.obs[domain_key] = np.nan
    full_adata.obs.loc[joint_batch_adata.obs_names, domain_key] = joint_batch_adata.obs[domain_key].values
    
    full_adata.obs['overlap'] = False
    full_adata.obs.loc[joint_batch_adata.obs_names, 'overlap'] = True
    
    # Define training (overlap) and prediction (non-overlap) masks
    train_mask = full_adata.obs['overlap'] == True
    predict_mask = full_adata.obs['overlap'] == False
    
    train_mask = train_mask & full_adata.obs[domain_key].notna()
    if sum(train_mask) == 0:
        raise ValueError(f"Batch {batch_id} has no valid training samples (overlap=True and with labels)")
    
    le = LabelEncoder()
    y_train_str = full_adata.obs[domain_key][train_mask].astype(str).values
    y_train = le.fit_transform(y_train_str)
    
    X_raw = full_adata.X[train_mask]
    X_train = X_raw.toarray() if sp.issparse(X_raw) else X_raw.copy()
    X_predict_raw = full_adata.X[predict_mask]
    X_predict = X_predict_raw.toarray() if sp.issparse(X_predict_raw) else X_predict_raw.copy()
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
    X_predict_processed = scaler.transform(imputer.transform(X_predict))
    
    pca = PCA(n_components=50 if full_adata.n_vars>50 else full_adata.n_vars-1, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_processed)
    X_predict_pca = pca.transform(X_predict_processed)

    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        objective='multi:softmax',
        num_class=len(le.classes_),
        random_state=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_jobs=-1,
        verbosity=0
    )
    
    cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=3, scoring='accuracy')

    # Train full model and predict
    clf.fit(X_train_pca, y_train)
    y_pred_encoded = clf.predict(X_predict_pca)
    y_pred = le.inverse_transform(y_pred_encoded)
    

    full_adata.obs.loc[predict_mask, domain_key] = y_pred
    train_acc = accuracy_score(y_train, clf.predict(X_train_pca))
    
    return full_adata





def identify_overlapping_regions_multi_slices(
    multiple_slices_dict, 
    distance_threshold=20,
    spatial_key='aligned_spatial'
):
    """
    Identify overlapping regions between two slices, generate new adata containing only overlapping obs for each modality of each slice
    
    Parameters
    ----------
    multiple_slices_dict : dict
        Format: {
            'slice1_name': {'modality1': adata1, 'modality2': adata2},
            'slice2_name': {'modality1': adata3, 'modality2': adata4}
        }
        Requirement: Exactly 2 slices, each with exactly 2 modalities (custom names allowed, e.g., rna/peak, rna/protein)
    distance_threshold : int/float
        Spatial distance threshold (same unit as spatial coordinates, default=20)
    spatial_key : str
        Key in obsm storing aligned spatial coordinates (default='aligned_spatial')
    
    Returns
    ----------
    multiple_slices_dict : dict
        Original dict with new adata for overlapping regions (keys: {modality_name}_overlap)
    """
    # Extract slice names (ensure 2 slices only)
    slice_names = list(multiple_slices_dict.keys())
    assert len(slice_names) == 2, "multiple_slices_dict must contain exactly 2 slices!"
    slice1_name, slice2_name = slice_names[0], slice_names[1]
    
    # Extract modality names (ensure 2 modalities per slice)
    slice1_modalities = list(multiple_slices_dict[slice1_name].keys())
    slice2_modalities = list(multiple_slices_dict[slice2_name].keys())
    assert len(slice1_modalities) == len(slice2_modalities) == 2, "Each slice must contain exactly 2 modalities!"
    
    # Get spatial coords (first modality of each slice, coords should be consistent per slice)
    adata_slice1 = multiple_slices_dict[slice1_name][slice1_modalities[0]]
    adata_slice2 = multiple_slices_dict[slice2_name][slice2_modalities[0]]
    ref_coords = adata_slice1.obsm[spatial_key].astype(np.float32)
    src_coords = adata_slice2.obsm[spatial_key].astype(np.float32)
    
    # KDTree neighbor search
    src_tree = cKDTree(src_coords)
    distances, src_indices = src_tree.query(ref_coords, distance_upper_bound=distance_threshold)
    
    # Calculate overlap masks (bidirectional validation)
    slice1_overlap_mask = distances < distance_threshold
    unique_slice2_indices = np.unique(src_indices[slice1_overlap_mask])
    slice1_overlap_mask = np.isin(src_indices, unique_slice2_indices)
    
    ref_tree = cKDTree(ref_coords)
    distances2, ref_indices2 = ref_tree.query(src_coords, distance_upper_bound=distance_threshold)
    slice2_overlap_mask = distances2 < distance_threshold
    unique_slice1_indices = np.unique(ref_indices2[slice2_overlap_mask])
    slice2_overlap_mask = np.isin(ref_indices2, unique_slice1_indices)
    
    # Overlap stats
    # print(f"===== Overlap Region Statistics =====")
    # print(f"Slice [{slice1_name}]: Total spots={len(slice1_overlap_mask)}, Overlap spots={np.sum(slice1_overlap_mask)}")
    # print(f"Slice [{slice2_name}]: Total spots={len(slice2_overlap_mask)}, Overlap spots={np.sum(slice2_overlap_mask)}")
    
    # Process slice1 modalities (generate overlap adata)
    for modality in slice1_modalities:
        adata = multiple_slices_dict[slice1_name][modality]
        if hasattr(adata, 'X'):
            x_sum = adata.X.sum(axis=1).flatten() if hasattr(adata.X, 'toarray') else adata.X.sum(axis=1)
            x_sum = np.array(x_sum).squeeze()  
            overlap_valid = slice1_overlap_mask & (x_sum > 0)
        else:
            overlap_valid = slice1_overlap_mask
        
        assert overlap_valid.ndim == 1 and len(overlap_valid) == adata.n_obs, f"{slice1_name}-{modality} mask dimension mismatch!"
        adata_overlap = adata[overlap_valid].copy()
        multiple_slices_dict[slice1_name][f"{modality}_overlap"] = adata_overlap
        # print(f"Slice [{slice1_name}]_[{modality}]: Valid overlap spots after filtering={np.sum(overlap_valid)}")
    
    # Process slice2 modalities (generate overlap adata)
    for modality in slice2_modalities:
        adata = multiple_slices_dict[slice2_name][modality]
        if hasattr(adata, 'X'):
            x_sum = adata.X.sum(axis=1).flatten() if hasattr(adata.X, 'toarray') else adata.X.sum(axis=1)
            x_sum = np.array(x_sum).squeeze()
            overlap_valid = slice2_overlap_mask & (x_sum > 0)
        else:
            overlap_valid = slice2_overlap_mask
        
        assert overlap_valid.ndim == 1 and len(overlap_valid) == adata.n_obs, f"{slice2_name}-{modality} mask dimension mismatch!"
        adata_overlap = adata[overlap_valid].copy()
        multiple_slices_dict[slice2_name][f"{modality}_overlap"] = adata_overlap
        # print(f"Slice [{slice2_name}]_[{modality}]: Valid overlap spots after filtering={np.sum(overlap_valid)}")
    
    return multiple_slices_dict
