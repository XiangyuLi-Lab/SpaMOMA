
        
        
def run_MISO(adata1,modality1,adata2,modality2):
    print('************ running MISO ***************')
    from .miso.hist_features import get_features
    from .miso.utils import preprocess
    from .miso import Miso

    import pandas as pd
    import numpy as np
    import scanpy as sc
    import torch
    import random

    from scipy.sparse import issparse

    # save current random seed
    np_seed_state = np.random.get_state()
    random_seed_state = random.getstate()
    torch_seed_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_seed_state = torch.cuda.get_rng_state_all()


    seed=100
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print("CUDA is not available. Using CPU.")


    if modality1 in ['rna','gene_activity_score','peak']:
        sc.pp.normalize_total(adata1)
    elif modality1 in ['protein']:
        adata1.X = preprocess(adata1,modality='protein')
    
    if modality2 in ['rna','gene_activity_score','peak']:
        sc.pp.normalize_total(adata2)
    elif modality2 in ['protein']:
        adata2.X = preprocess(adata2,modality='protein')
        
    adata_list=[adata1,adata2]

    features=[]
    for adata in adata_list:
        if issparse(adata.X):
            mat = adata.X.toarray()
        else:
            mat = adata.X.copy()  
        
        features.append(mat)
        
    model = Miso(features,ind_views='all',combs='all',sparse=False,device=device)
    model.train()
    
    # restore previous random seed
    np.random.set_state(np_seed_state)
    random.setstate(random_seed_state)
    torch.set_rng_state(torch_seed_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_seed_state)
    
    return model.emb


