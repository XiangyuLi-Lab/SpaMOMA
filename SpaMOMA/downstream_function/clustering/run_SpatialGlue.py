
import os

import torch
import pandas as pd
import scanpy as sc
import os


from .SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi


# %%
def preprocess_adata(adata,modality):
    print(f"processing {modality}")
    if modality in ['rna','gene_activity_score']:
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        adata_high =  adata[:, adata.var['highly_variable']]
        adata.obsm['feat'] = pca(adata_high, n_comps=50)
    elif modality=='peak':
        if 'X_lsi' not in adata.obsm.keys():
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata, use_highly_variable=False, n_components=51)
        adata.obsm['feat'] = adata.obsm['X_lsi'].copy()
    elif modality=='protein':
        
        adata = clr_normalize_each_cell(adata)
        sc.pp.scale(adata)
        adata.obsm['feat'] = pca(adata, n_comps=adata.n_vars-1)
    
    return adata

def run_SpatialGlue(adata1,modality1,adata2,modality2,data_type):
    print('************ running SpatialGlue ***************')
    import numpy as np
    import random
    # save current random seed
    np_seed_state = np.random.get_state()
    random_seed_state = random.getstate()
    torch_seed_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_seed_state = torch.cuda.get_rng_state_all()
        
        
    from .SpatialGlue.preprocess import fix_seed
    random_seed = 2025
    fix_seed(random_seed)
    device='cuda'

    adata1=preprocess_adata(adata1,modality1)
    adata2=preprocess_adata(adata2,modality2)

    adata1.obsm['spatial']=adata1.obsm['aligned_spatial'].copy()
    adata2.obsm['spatial']=adata2.obsm['aligned_spatial'].copy()
    
    if modality1=='rna':
        adata_omics1=adata1.copy()
        adata_omics2=adata2.copy()
    else:
        adata_omics1=adata2.copy()
        adata_omics2=adata1.copy()

    from .SpatialGlue.preprocess import construct_neighbor_graph
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

    # define model
    from .SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
    model = Train_SpatialGlue(data, datatype=data_type, device=device)

    # train model
    output = model.train()
    
    # restore previous random seed
    np.random.set_state(np_seed_state)
    random.setstate(random_seed_state)
    torch.set_rng_state(torch_seed_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_seed_state)

    return output['SpatialGlue']




