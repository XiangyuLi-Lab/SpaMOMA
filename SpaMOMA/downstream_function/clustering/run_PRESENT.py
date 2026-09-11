import os
import torch
import anndata as ad
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp

from .PRESENT import PRESENT_function


def convert_to_non_negative_int(adata):
    adata.layers["before_int_convert"] = adata.X.copy()

    if sp.issparse(adata.X):
        adata.X.data = np.round(adata.X.data).astype(np.int32)
        adata.X.data[adata.X.data < 0] = 0
        adata.X.eliminate_zeros()
    else:
        adata.X = np.round(adata.X).astype(np.int32)
        adata.X[adata.X < 0] = 0

    return adata


def run_PRESENT(adata1,modality1,adata2,modality2):
    
    import torch
    import random
    # save current random seed
    np_seed_state = np.random.get_state()
    random_seed_state = random.getstate()
    torch_seed_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_seed_state = torch.cuda.get_rng_state_all()
    

    from .PRESENT import PRESENT_function
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if modality1=='rna' or modality1=='gene_activity_score':
        adata_rna=adata1
        adata_rna = convert_to_non_negative_int(adata_rna)
    elif modality1=='peak':
        adata_atac=adata1
        adata_atac = convert_to_non_negative_int(adata_atac)
    elif modality1=='protein':
        adata_adt=adata1
        
    if modality2=='rna' or modality2=='gene_activity_score':
        adata_rna=adata2
        adata_rna = convert_to_non_negative_int(adata_rna)
    elif modality2=='peak':
        adata_atac=adata2
        adata_atac = convert_to_non_negative_int(adata_atac)
    elif modality2=='protein':
        adata_adt=adata2
        

    if {modality1,modality2}=={'rna','peak'} or {modality1,modality2}=={'gene_activity_score','peak'}:
        adata = PRESENT_function(
            spatial_key='spatial',
            adata_rna = adata_rna,
            adata_atac = adata_atac
        )
    elif {modality1,modality2}=={'rna','protein'}:
        adata = PRESENT_function(
            spatial_key='spatial',
            adata_rna = adata_rna,
            adata_adt = adata_adt
        )
        
    # restore previous random seed
    np.random.set_state(np_seed_state)
    random.setstate(random_seed_state)
    torch.set_rng_state(torch_seed_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_seed_state)

    return adata

def run_PRESENT(adata1,modality1,adata2,modality2):
    print('************ running PRESENT ***************')
    
    import torch
    import random
    # save current random seed
    np_seed_state = np.random.get_state()
    random_seed_state = random.getstate()
    torch_seed_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_seed_state = torch.cuda.get_rng_state_all()
    

    
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if modality1=='rna' or modality1=='gene_activity_score':
        adata_rna=adata1
        adata_rna = convert_to_non_negative_int(adata_rna)
    elif modality1=='peak':
        adata_atac=adata1
        adata_atac = convert_to_non_negative_int(adata_atac)
    elif modality1=='protein':
        adata_adt=adata1
        
    if modality2=='rna' or modality2=='gene_activity_score':
        adata_rna=adata2
        adata_rna = convert_to_non_negative_int(adata_rna)
    elif modality2=='peak':
        adata_atac=adata2
        adata_atac = convert_to_non_negative_int(adata_atac)
    elif modality2=='protein':
        adata_adt=adata2
        

    if {modality1,modality2}=={'rna','peak'} or {modality1,modality2}=={'gene_activity_score','peak'}:
        adata = PRESENT_function(
            spatial_key='spatial',
            adata_rna = adata_rna,
            adata_atac = adata_atac
        )
    elif {modality1,modality2}=={'rna','protein'}:
        adata = PRESENT_function(
            spatial_key='spatial',
            adata_rna = adata_rna,
            adata_adt = adata_adt
        )
        
    # restore previous random seed
    np.random.set_state(np_seed_state)
    random.setstate(random_seed_state)
    torch.set_rng_state(torch_seed_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_seed_state)

    return adata







def run_PRESENT_multiple_slices(multiple_slices_dict,batch_key='batch',spatial_key='aligned_spatial'):
    # multiple_slices_dict=
    # {sample_id: 
    #   {'RNA': full_adata, 
    #    'ATAC': full_adata,
    #    'RNA_overlap':overlap_adata,
    #    'ATAC_overlap':overlap_adata
    #   }
    # }
    adata_omics1_list=[]
    adata_omics2_list=[]
    modality_list = list(multiple_slices_dict[list(multiple_slices_dict.keys())[0]].keys())
    modality1=modality_list[0]
    modality2=modality_list[1]
    for sample_id in multiple_slices_dict.keys():
        adata_dict_one_sample=multiple_slices_dict[sample_id]

        adata_overlap1=adata_dict_one_sample[f"{modality1}_overlap"]
        adata_overlap1=convert_to_non_negative_int(adata_overlap1)
        adata_omics1_list.append(adata_overlap1)
        
        adata_overlap2=adata_dict_one_sample[f"{modality2}_overlap"]
        adata_overlap2=convert_to_non_negative_int(adata_overlap2)
        adata_omics2_list.append(adata_overlap2)
        
    adata_omics1=ad.concat(adata_omics1_list)
    adata_omics2=ad.concat(adata_omics2_list)
    
    if modality1=='rna' or modality1=='gene_activity_score':
        adata_rna=adata_omics1
    elif modality1=='peak':
        adata_atac=adata_omics1
    elif modality1=='protein':
        adata_adt=adata_omics1
        
    if modality2=='rna' or modality2=='gene_activity_score':
        adata_rna=adata_omics2
    elif modality2=='peak':
        adata_atac=adata_omics2
    elif modality2=='protein':
        adata_adt=adata_omics2
    
    if {modality1,modality2} == {'rna','peak'} or {modality1,modality2}=={'gene_activity_score','peak'}:
        adata = PRESENT_function(
            spatial_key = spatial_key,
            adata_rna = adata_rna,
            adata_atac = adata_atac,
            batch_key = batch_key
        )
    elif {modality1,modality2}=={'rna','protein'}:
        adata = PRESENT_function(
            spatial_key = spatial_key,
            adata_rna = adata_rna,
            adata_adt = adata_adt,
            batch_key = batch_key
        )
    
    # labeled_full_adata_dict = {}  

    # for sample_id,transfer_reference_omics in reference_modality_dict.items():
 
    #     full_adata = multiple_slices_dict[sample_id][transfer_reference_omics].copy()

    #     full_adata_with_label = domain_label_transfer(
    #         batch_id=sample_id,
    #         full_adata=full_adata,
    #         joint_cluster_adata=adata,
    #         spatial_key=spatial_key,
    #         seed=0
    #     )
    #     labeled_full_adata_dict[sample_id] = full_adata_with_label
        
    # return labeled_full_adata_dict
    adata.obsm['PRESENT_emb']=adata.obsm['embeddings'].copy()
    return adata