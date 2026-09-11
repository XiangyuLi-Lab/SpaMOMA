
import os
import multiprocessing
import squidpy as sq
import scanpy as sc
import anndata as ad
import numpy as np
from scipy import sparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import anndata
from PIL import Image
import os
import cv2
import colorsys
from .utils.generate_domain_pic_tools import *
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import sys
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import STAGATE
from .test_time_adaption import run_adaptation
import subprocess
import signal
import json

import os
import sys


from .utils.match_utils import *
from .utils.match_visual import *
from .downstream_function.translation import translator 

import random
random.seed(42)  
np.random.seed(42)  
cv2.setRNGSeed(42)  
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)  


class SpaMOMA:

    def __init__(self,save_dir):
        self.input_slice_dict = {}
        self.save_dir=save_dir
        self.reference_sample_id=None
        self.fine_epochs=None
        self.coarse_epochs=None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def add_slice(self,
                  sample_id,
                  raw_data_path,
                  feature):
        
        if feature not in ['rna','gene_activity_score','peak','protein','metabolite','img']:
            raise ValueError(f"Invalid feature type {feature}!")
        
        if len(self.input_slice_dict.keys())==0:
            self.reference_sample_id=sample_id
            
        slice_info = {
            'sample_id': sample_id,
            'raw_data_path': raw_data_path,
            'feature': feature,
        }
        self.input_slice_dict[sample_id]=slice_info

        
    def set_reference(self,sample_id):
        self.reference_sample_id=sample_id
        
    
    def _process_omics_slice(self,sample_id,svf_num=2000,hvf_num=3000,pseudo_color_key='pca',k_cutoff=7,graph_mode='KNN',rad_cutoff=150):
        info=self.input_slice_dict[sample_id]
        self.input_slice_dict[sample_id]['pseudo_color_key']=pseudo_color_key
        feature=info['feature']
        raw_data_path=info['raw_data_path']
        adata=sc.read_h5ad(raw_data_path)
        print(f"preprocessing {feature} features...")
        if feature=='metabolite':
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvf_num)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            sq.gr.spatial_neighbors(adata, key_added='spatial')
            sq.gr.spatial_autocorr(adata, mode="moran")
            svg_list = adata.uns['moranI']['I'].nlargest(100).index.tolist()

            combined_genes = list(set(svg_list))
            combined_genes = [gene for gene in adata.var.index if gene in combined_genes]
            adata = adata[:, combined_genes]

            tfidf = TfidfTransformer()
            adata.X = tfidf.fit_transform(adata.X)
            
            svd_num=100

            svd = TruncatedSVD(n_components=svd_num, random_state=42)
            new_mat = svd.fit_transform(adata.X)
            new_mat = csr_matrix(new_mat)
            
            adata_temp = sc.AnnData(
                X=new_mat,
                obs=adata.obs,
                obsm=adata.obsm,
                uns=adata.uns,
                var=pd.DataFrame(index=[f"PC{i+1}" for i in range(svd_num)])  
            )


        elif feature=='rna' or feature=='gene_activity_score':
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvf_num)
            
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            
            sq.gr.spatial_neighbors(adata, key_added='spatial')
            sq.gr.spatial_autocorr(adata, mode="moran")
            svg_list = adata.uns['moranI']['I'].nlargest(svf_num).index.tolist()
            hvg_list = adata.var[adata.var['highly_variable']].index.tolist()
            combined_genes = list(set(hvg_list) | set(svg_list))
            
            combined_genes = [gene for gene in adata.var.index if gene in combined_genes]
            adata = adata[:, combined_genes]
            adata_temp=adata.copy()

        elif feature=='protein':
            adata_temp=adata.copy()
        

        elif feature=='peak':

            tfidf = TfidfTransformer()
            adata.X = tfidf.fit_transform(adata.X)
            
            svd = TruncatedSVD(n_components=100, random_state=42)
            new_mat = svd.fit_transform(adata.X)
            new_mat = csr_matrix(new_mat)

            adata_temp = sc.AnnData(
                X=new_mat,
                obs=adata.obs,
                obsm=adata.obsm,
                uns=adata.uns,
                var=pd.DataFrame(index=[f"PC{i+1}" for i in range(100)])  
            )
            
        if graph_mode=='KNN':
            STAGATE.Cal_Spatial_Net(adata_temp, model=graph_mode,k_cutoff=k_cutoff)
        elif graph_mode=='Radius':
            STAGATE.Cal_Spatial_Net(adata_temp, rad_cutoff=rad_cutoff)
        STAGATE.Stats_Spatial_Net(adata_temp)
        adata_temp = STAGATE.train_STAGATE(adata_temp, alpha=0,hidden_dims=[512,30])
        adata_temp = sc.AnnData(adata_temp.obsm['STAGATE'])
        sc.pp.pca(adata_temp, n_comps=3)
        adata.obsm[pseudo_color_key] = adata_temp.obsm['X_pca']
    
        return adata
    
    def _downsample_img(self,sample_id,n_clusters=5,gaussian_kernel=(15,15)):
        raw_img_path = self.input_slice_dict[sample_id]['raw_data_path']
        
        downsampled_img = image_to_color_blocks(
            img_path=raw_img_path,
            n_clusters=n_clusters,
            gaussian_kernel=gaussian_kernel,
            show_result=True
        )
        self.input_slice_dict[sample_id]['processed_data']=downsampled_img
    
    def preprocess_input_slices(self,params_dict={}):
        

        valid_sample_ids = set(self.input_slice_dict.keys())
        provided_sample_ids = set(params_dict.keys())
        invalid_sample_ids = provided_sample_ids - valid_sample_ids
        if invalid_sample_ids:
            raise ValueError(
                f"Invalid sample IDs provided: {invalid_sample_ids}. "
                f"Valid sample IDs are: {valid_sample_ids}"
            )

        for sample_id in self.input_slice_dict.keys():
            
            feature=self.input_slice_dict[sample_id]['feature']
            if feature=='img':
                valid_params = ['gaussian_kernel', 'n_clusters']
                sample_params = params_dict.get(sample_id, {})
                filtered_downsample_params = {
                    k: v for k, v in sample_params.items() 
                    if k in valid_params
                }
                if sample_params.get('downsample', False)==True:
                    self._downsample_img(sample_id,**filtered_downsample_params)
                    
                    
            elif feature in ['rna','gene_activity_score','peak','protein','metabolite']:
                valid_params = ['svf_num', 'hvf_num', 'pseudo_color_key', 'k_cutoff', 'graph_mode', 'rad_cutoff']
                sample_params = params_dict.get(sample_id, {})
                
                filtered_params = {k: v for k, v in sample_params.items() if k in valid_params}
                processed_adata=self._process_omics_slice(sample_id,**filtered_params)
                self.input_slice_dict[sample_id]['processed_data']=processed_adata
                if not os.path.exists(f"{self.save_dir}/{sample_id}/"):
                    os.makedirs(f"{self.save_dir}/{sample_id}/")
                processed_adata.obs['batch']=sample_id
                processed_adata.write(f'{self.save_dir}/{sample_id}/processed_data.h5ad')
        
        
    def generate_imgs(self,params_dict={}):
        valid_sample_ids = set(self.input_slice_dict.keys())
        provided_sample_ids = set(params_dict.keys())
        invalid_sample_ids = provided_sample_ids - valid_sample_ids
        if invalid_sample_ids:
            raise ValueError(
                f"Invalid sample IDs provided: {invalid_sample_ids}. "
                f"Valid sample IDs are: {valid_sample_ids}"
            )
        
        valid_visualize_params = [
            'max_distance_ratio', 'interpolation_method','gaussian_sigma'
        ]
        
        for sample_id in self.input_slice_dict.keys():
            os.makedirs(f'{self.save_dir}/{sample_id}/',exist_ok=True)
            
            data_info=self.input_slice_dict[sample_id]
            feature=data_info['feature']
            sample_params = params_dict.get(sample_id, {})
            if feature=='img':
                if 'processed_data' in data_info:
                    downsample_img=data_info['processed_data']
                    img_save = cv2.cvtColor(downsample_img, cv2.COLOR_RGB2BGR)
                    save_path = f'{self.save_dir}/{sample_id}/RGB.png'
                    cv2.imencode('.png', img_save)[1].tofile(save_path)  
                    self.input_slice_dict[sample_id]['SPI_path']=os.path.abspath(save_path)
                    print(f"Color-quantized image saved to: {save_path}")
                else:
                    raw_img_path = self.input_slice_dict[sample_id]['raw_data_path']
                    raw_img = cv2.imread(raw_img_path)
                    if raw_img is None:
                        raw_img = Image.open(raw_img_path).convert('RGB')
                        raw_img = np.array(raw_img)
                        img_save = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                    else:
                        img_save = raw_img
                    save_path = f'{self.save_dir}/{sample_id}/RGB.png'
                    cv2.imencode('.png', img_save)[1].tofile(save_path)
                    
                    self.input_slice_dict[sample_id]['SPI_path']=os.path.abspath(save_path)
                    print(f"Original image saved to: {self.input_slice_dict[sample_id]['SPI_path']}")
            elif feature in ['rna','gene_activity_score','peak','protein','metabolite']:
                processed_adata=data_info['processed_data']
                pseudo_color_key=data_info['pseudo_color_key']
    
                filtered_visualize_params = {
                    k: v for k, v in sample_params.items() 
                    if k in valid_visualize_params
                }
                img, path = visualize_RGB(
                    processed_adata,
                    rgb_key=pseudo_color_key,
                    output_dir=f'{self.save_dir}/{sample_id}/',
                    **filtered_visualize_params
                )
                processed_adata.write(f'{self.save_dir}/{sample_id}/processed_data.h5ad')
                self.input_slice_dict[sample_id]['SPI_path']=os.path.abspath(path)
                print(f"Img for {sample_id} is saved to {self.input_slice_dict[sample_id]['SPI_path']}")
    
    

    

    def train(self,coarse_epochs=1000,fine_epochs=300,batch_size=4,
        parallel_mode=False, 
        coarse_checkpoint_save_step=None,
        fine_checkpoint_save_step=None,
        coarse_gpu=3,         
        fine_gpu=4):     
        
        self.coarse_epochs=coarse_epochs
        self.fine_epochs=fine_epochs

       
        img_paths_coarse = [
            self.input_slice_dict[self.reference_sample_id]['SPI_path']
        ]
        epoch_angle_config_coarse = []

        if coarse_epochs >= 500:
            coarse_stages = [
                (0, 100, [0, 45]),
                (101, 200, [0, 45, 90]),
                (201, 300, [0, 45, 90, 135, 180]),
                (351, 400, [0, 45, 90, 135, 180, 225]),
                (401, 500, [0, 45, 90, 135, 180, 225, 270]),
                (501, coarse_epochs, [0, 45, 90, 135, 180, 225, 270, 315]),
            ]

            for start, end, angles in coarse_stages:
                if start > coarse_epochs:
                    break
                epoch_angle_config_coarse.append({
                    "start": start,
                    "end": min(end, coarse_epochs),
                    "angles": angles * batch_size
                })

        else:
            stage_epochs = (coarse_epochs + 5) // 6
            start_epoch = 0

            angle_lists = [
                [0, 45],
                [0, 45, 90],
                [0, 45, 90, 135, 180],
                [0, 45, 90, 135, 180, 225],
                [0, 45, 90, 135, 180, 225, 270],
                [0, 45, 90, 135, 180, 225, 270, 315]
            ]

            for angles in angle_lists:
                if start_epoch > coarse_epochs:
                    break

                end_epoch = min(start_epoch + stage_epochs, coarse_epochs)

                epoch_angle_config_coarse.append({
                    "start": start_epoch,
                    "end": end_epoch,
                    "angles": angles * batch_size
                })

                start_epoch = end_epoch + 1

        coarse_config = {
            "img_paths": img_paths_coarse,
            "epochs": coarse_epochs,
            "batch_size": batch_size,
            "checkpoint_dir": f"{self.save_dir}/coarse_model/",
            "epoch_angle_config": epoch_angle_config_coarse,
            'checkpoint_save_step':coarse_checkpoint_save_step
        }
      
        img_paths_fine = []
        for sample_id,data_info in self.input_slice_dict.items():
            img_paths_fine.append(data_info['SPI_path'])
        
        epoch_angle_config_fine = [
            {
                "start": 0,
                "end": fine_epochs,
                "angles": [0] * batch_size
            }
        ]

        fine_config = {
            "img_paths": img_paths_fine,
            "epochs": fine_epochs,
            "batch_size": batch_size,
            "checkpoint_dir": f"{self.save_dir}/fine_model/",
            "epoch_angle_config": epoch_angle_config_fine,
            'checkpoint_save_step':fine_checkpoint_save_step
        }


        # p1 = subprocess.Popen(
        #     ["python", "run_coarse_adaptation.py", json.dumps(coarse_config)],
        #     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(coarse_gpu)}
        # )

        # p2 = subprocess.Popen(
        #     ["python", "run_fine_adaptation.py", json.dumps(fine_config)],
        #     env={**os.environ, "CUDA_VISIBLE_DEVICES": str(fine_gpu)}
        # )

        # p1.wait()
        # p2.wait()
            
            
        coarse_env = os.environ.copy()
        coarse_env["CUDA_VISIBLE_DEVICES"] = str(coarse_gpu)

        fine_env = os.environ.copy()
        fine_env["CUDA_VISIBLE_DEVICES"] = str(fine_gpu)


        coarse_log_path = f"{self.save_dir}/coarse_adapt.log"
        fine_log_path = f"{self.save_dir}/fine_adapt.log"
        coarse_log_file = open(coarse_log_path, "w")
        fine_log_file = open(fine_log_path, "w")
        
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        coarse_env["PYTHONPATH"] = project_root
        fine_env["PYTHONPATH"] = project_root
        coarse_script_path = os.path.join(current_file_dir, "run_coarse_adaptation.py")
        fine_script_path = os.path.join(current_file_dir, "run_fine_adaptation.py")


        def _launch_process(cmd, env, log_file):
            return subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid 
            )
        
        p1 = None
        p2 = None
        try:

            if parallel_mode:
                
                p1 = _launch_process(
                    ["python", '-m',"SpaMOMA.run_coarse_adaptation", json.dumps(coarse_config)],
                    coarse_env,
                    coarse_log_file
                )
                print('start coarse adaptation...')

                p2 = _launch_process(
                    ["python", '-m',"SpaMOMA.run_fine_adaptation", json.dumps(fine_config)],
                    fine_env,
                    fine_log_file
                )
                print('start fine adaptation...')


                p1.wait()
                p2.wait()
                print('fine adaptation done!')
                print('coarse adaptation done!')

            else:

                p1 = _launch_process(
                    ["python", coarse_script_path, json.dumps(coarse_config)],
                    coarse_env,
                    coarse_log_file
                )
                p1.wait()

                p2 = _launch_process(
                    ["python", fine_script_path, json.dumps(fine_config)],
                    fine_env,
                    fine_log_file
                )
                p2.wait()
        
        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Killing child processes...")
            for p in [p1, p2]:
                if p is not None and p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        p.wait(timeout=2)
                    except:
                        try:
                            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                        except:
                            pass
            raise

        finally:
            coarse_log_file.close()
            fine_log_file.close()


    def load_images(self,image_path1, image_path2):

        image0 = cv2.imread(image_path1)
        image1 = cv2.imread(image_path2)
        
        if image0 is None or image1 is None:
            raise FileNotFoundError("Failed to load images, check file paths")
        
        image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        return image0, image1

    def align_slices(self,coarse_weights_path=None,fine_weights_path=None,visualize_correspondences=False):

        if not os.path.exists(f'{self.save_dir}/alignment_output/'):
            os.makedirs(f'{self.save_dir}/alignment_output/')
 
        sample_id_ref = self.reference_sample_id
        sample_id_src_list = [sid for sid in self.input_slice_dict.keys() if sid != sample_id_ref]

        if not sample_id_src_list:
            raise ValueError(f"sample_id_src_list is empty after filtering reference ID '{sample_id_ref}'! Check if input_slice_dict has other samples.")

        matcher_zoo={}
        
        if coarse_weights_path is None:
            import glob
            coarse_dir = f'{self.save_dir}/coarse_model/'
            coarse_files = glob.glob(os.path.join(coarse_dir, '*'))
            coarse_files = [f for f in coarse_files if os.path.isfile(f)]
            if coarse_files:
                coarse_weights_path = coarse_files[0]
            else:
                raise FileNotFoundError(f"No coarse model files found in directory {coarse_dir}")
        

        if fine_weights_path is None:
            import glob
            fine_dir = f'{self.save_dir}/fine_model/'
            fine_files = glob.glob(os.path.join(fine_dir, '*'))
            fine_files = [f for f in fine_files if os.path.isfile(f)]
            if fine_files:
                fine_weights_path = fine_files[0]
            else:
                raise FileNotFoundError(f"No fine model files found in directory {fine_dir}")

        params = setup_matching_parameters()


        if self.input_slice_dict[self.reference_sample_id]['feature']!='img':
            adata_ref=sc.read_h5ad(f'{self.save_dir}/{sample_id_ref}/processed_data.h5ad')
            
            adata_ref_raw=sc.read_h5ad(self.input_slice_dict[sample_id_ref]['raw_data_path'])
            adata_save=sc.AnnData(X=adata_ref_raw.X,
                                var=adata_ref_raw.var,
                                obs=adata_ref.obs,
                                obsm=adata_ref.obsm,
                                uns=adata_ref.uns)
            
            adata_save.obsm['aligned_spatial']=adata_ref.obs[['my_pixel_x','my_pixel_y']].values
            
            ref_adata_save_path=f"{self.save_dir}/alignment_output/{sample_id_ref}.h5ad"
            adata_save.write(ref_adata_save_path)
            print(f"aligned {sample_id_ref} adata is saved to: {ref_adata_save_path}")
        else:
            ref_img_path = f"{self.save_dir}/{sample_id_ref}/RGB.png"
            ref_img = cv2.imread(ref_img_path)
            if ref_img is None:
                ref_img = Image.open(ref_img_path).convert('RGB')
                ref_img = np.array(ref_img)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
            ref_save_path = f"{self.save_dir}/alignment_output/{sample_id_ref}.png"
            cv2.imencode('.png', ref_img)[1].tofile(ref_save_path)  
            print(f"{sample_id_ref} image saved to: {ref_save_path}")

        for sample_id_src in sample_id_src_list:
            image_path1 = f"{self.save_dir}/{sample_id_ref}/RGB.png"  
            image_path2 = f"{self.save_dir}/{sample_id_src}/RGB.png"

            image0, image1 = self.load_images(image_path1, image_path2)
            
            print(f'****************** aligning {sample_id_src} to {sample_id_ref} ..... ******************')
            print(f"start coarse alignment...")
            matcher_zoo['roma']={
                'matcher': {
                    'output': 'matches-roma',
                    'model': {
                    'name': 'roma',
                    'weights': 'outdoor',
                    'model_name': 'roma_outdoor.pth',
                    'weights_path': coarse_weights_path  
                    },
                    'preprocessing': {
                    'grayscale': False,
                    'force_resize': True,
                    'resize_max': 1024,
                    'width': 560,
                    'height': 560,
                    'dfactor': 8
                    }
                    },
                'dense': True
            }

            print('match params:',params)
            results = run_matching(
                image0=image0, 
                image1=image1, 
                keypoint_threshold=0.99,
                key=params["model_key"],
                match_threshold=params["match_threshold"],
                extract_max_keypoints=params["extract_max_keypoints"],
                force_resize=params["force_resize"],
                image_height=params["image_height"],
                image_width=params["image_width"],
                choice_geometry_type=params["choice_geometry_type"],
                matcher_zoo=matcher_zoo
            )

            if visualize_correspondences==True:
                visualize_matching_results(results)

            # correspondences (N, 2) and confidence (N,)
            kpts0 = results['state_cache']['keypoints0_orig'] 
            kpts1 = results['state_cache']['keypoints1_orig']  
            mconf = results['state_cache']['mconf']   

            conf_threshold = 0.0

            # filter high-confidence correspondences
            mask = mconf > conf_threshold
            valid_kpts0 = kpts0[mask]  
            valid_kpts1 = kpts1[mask]  

            if len(valid_kpts0) < 3:
                raise ValueError(f"Less than 3 valid keypoint pairs (only {len(valid_kpts0)}), cannot estimate affine transformation")


            M, inliers = cv2.estimateAffine2D(
                from_=valid_kpts1,
                to=valid_kpts0,
                method=cv2.RANSAC,
                ransacReprojThreshold=10.0,  
                maxIters=10000,             
                confidence=0.999            
            )

            if M is None:
                raise RuntimeError("Failed to estimate affine transformation matrix, check keypoint pairs")

            h, w = image0.shape[:2]
            image1_aligned = cv2.warpAffine(
                src=image1,    
                M=M,             
                dsize=(w, h),     
                flags=cv2.INTER_LINEAR,  
                borderMode=cv2.BORDER_CONSTANT,  
                borderValue=(0, 0, 0) 
            )

            print(f"coarse alignment done!")
            print(f"start fine alignment...")

            matcher_zoo['roma']={
                'matcher': {
                    'output': 'matches-roma',
                    'model': {
                    'name': 'roma',
                    'weights': 'outdoor',
                    'model_name': 'roma_outdoor.pth',
                    'weights_path': fine_weights_path  
                    },
                    'preprocessing': {
                    'grayscale': False,
                    'force_resize': True,
                    'resize_max': 1024,
                    'width': 560,
                    'height': 560,
                    'dfactor': 8
                    }
                    },
                'dense': True
            }


            results_fine = run_matching(
                image0=image0,  
                image1=image1_aligned,  
                keypoint_threshold=0.99,
                key=params["model_key"],
                match_threshold=params["match_threshold"],  
                extract_max_keypoints=params["extract_max_keypoints"],
                force_resize=params["force_resize"],
                image_height=params["image_height"],
                image_width=params["image_width"], 
                choice_geometry_type=params["choice_geometry_type"],
                matcher_zoo=matcher_zoo
            )

            if visualize_correspondences==True:
                visualize_matching_results(results_fine)

            kpts0_fine = results_fine['state_cache']['keypoints0_orig']
            kpts1_fine = results_fine['state_cache']['keypoints1_orig'] 


            mask_bg_texture = filter_background_texture_points(
                kpts0=kpts0_fine,
                kpts1=kpts1_fine,
                image0=image0,  
                image1_aligned=image1_aligned,  
                mean_thresh=0.1,  
                var_thresh=2e-4,
                patch_size=25       
            )


            # filter valid correspondence
            valid_kpts0_fine = kpts0_fine[mask_bg_texture]
            valid_kpts1_fine = kpts1_fine[mask_bg_texture]

            if visualize_correspondences==True:
                visualize_filtered_matches(
                    image0=image0,
                    image1_aligned=image1_aligned,
                    kpts0=valid_kpts0_fine,
                    kpts1=valid_kpts1_fine,
                    point_size=3,
                    line_alpha=0.6
                )

            if len(valid_kpts0_fine) < 3:
                raise ValueError(f"Less than 3 valid keypoint pairs for fine alignment (only {len(valid_kpts0_fine)}), cannot estimate transformation matrix")


            M_fine, inliers_fine = cv2.estimateAffine2D(
                from_=valid_kpts1_fine, 
                to=valid_kpts0_fine,    
                method=cv2.RANSAC,
                ransacReprojThreshold=10.0,  
                maxIters=10000,
                confidence=0.999
            )

            if M_fine is None:
                raise RuntimeError("Failed to estimate affine transformation matrix for fine alignment")


            image1_final = cv2.warpAffine(
                src=image1_aligned,  
                M=M_fine,           
                dsize=(w, h),      
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            image1_final_vis = image1_final.astype(np.float32) / 255.0
            mask_final = (image1_final_vis == [0, 0, 0]).all(axis=-1)
            image1_final_vis[mask_final] = [1, 1, 1]


            def compose_affine(M1, M2):
                M1_hom = np.vstack([M1, [0, 0, 1]])
                M2_hom = np.vstack([M2, [0, 0, 1]])
                T_hom = np.dot(M2_hom, M1_hom)
                return T_hom[:2, :]

            T_total = compose_affine(M, M_fine)

            if self.input_slice_dict[sample_id_src]['feature']!='img':
                adata_src=sc.read_h5ad(f'{self.save_dir}/{sample_id_src}/processed_data.h5ad')
                points_original = adata_src.obs[['my_pixel_x', 'my_pixel_y']].values.astype(np.float32)
                points_homogeneous = np.hstack([points_original, np.ones((points_original.shape[0], 1), dtype=np.float32)])
                points_final = np.dot(points_homogeneous, T_total.T)  
                
                adata_src_raw=sc.read_h5ad(self.input_slice_dict[sample_id_src]['raw_data_path'])
                adata_save=sc.AnnData(X=adata_src_raw.X,
                                      var=adata_src_raw.var,
                                      obs=adata_src.obs,
                                      obsm=adata_src.obsm,
                                      uns=adata_src.uns)
                
                adata_save.obsm['aligned_spatial']=points_final.copy()
                src_adata_save_path=f"{self.save_dir}/alignment_output/{sample_id_src}.h5ad"
                adata_save.write(src_adata_save_path)
                print(f"aligned {sample_id_src} adata is saved to: {src_adata_save_path}")
            else:
                src_img_path = f"{self.save_dir}/{sample_id_src}/RGB.png"
                src_img = cv2.imread(src_img_path)
                if src_img is None:
                    src_img = Image.open(src_img_path).convert('RGB')
                    src_img = np.array(src_img)
                    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                
                ref_h, ref_w = image0.shape[:2]
                src_img_aligned = cv2.warpAffine(
                    src=src_img,
                    M=T_total,
                    dsize=(ref_w, ref_h),  
                    flags=cv2.INTER_LINEAR,  
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                
                src_save_path = f"{self.save_dir}/alignment_output/{sample_id_src}.png"
                cv2.imencode('.png', src_img_aligned)[1].tofile(src_save_path)  
                print(f"aligned {sample_id_src} image is saved to: {src_save_path}")
            
            print(f"fine alignment done!")
            
            image0_vis = image0
            image1_vis = image1
            image1_aligned_vis = image1_aligned.astype(np.float32) / 255.0
            mask = (image1_aligned_vis == [0, 0, 0]).all(axis=-1)
            image1_aligned_vis[mask] = [1, 1, 1]
            if visualize_correspondences==True:
                plt.figure(figsize=(24, 6))

                plt.subplot(152)
                plt.imshow(image1_vis)
                plt.title("Original Src", fontsize=12)
                plt.axis('off')

                plt.subplot(151)
                plt.imshow(image0_vis)
                plt.title("Reference", fontsize=12)
                plt.axis('off')

                plt.subplot(153)
                plt.imshow(image1_aligned_vis)
                plt.title("Coarse Alignment", fontsize=12)
                plt.axis('off')

                plt.subplot(154)
                plt.imshow(image1_final_vis)
                plt.title("Fine Alignment (Final)", fontsize=12)
                plt.axis('off')

                plt.subplot(155)
                overlay = cv2.addWeighted(image0_vis.astype(np.float32)/255.0, 0.5, image1_final_vis, 0.5, 0)
                plt.imshow(overlay)
                plt.title("Overlay (Reference + Final Src)", fontsize=12)
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  
                torch.cuda.synchronize() 
            random.seed(42)
            np.random.seed(42)
            cv2.setRNGSeed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
            plt.close('all')  
        
    
    def read_aligned_data(self,sample_id):
        if self.input_slice_dict[sample_id]['feature']=='img':
            img_path = f'{self.save_dir}/alignment_output/{sample_id}.png'
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Aligned image file not found for sample {sample_id} at path: {img_path}")
            
            img = cv2.imread(img_path)
            if img is None:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            data_path=f'{self.save_dir}/alignment_output/{sample_id}.h5ad'
            if not os.path.exists(data_path):
                raise ValueError("h5ad not exist!")
            return sc.read_h5ad(data_path)
    

    def visualize_aligned_slices(self, figsize=(4, 4), point_size=1, alpha_adata=0.7, alpha_img=0.9, 
                                color_map=None, save_fig=False, dpi=300):
        
        if color_map is None:
            default_colors = ['steelblue','lightcoral',  'forestgreen', 'purple', 'orange', 'gold']
            sample_ids = list(self.input_slice_dict.keys())
            color_map = {sid: default_colors[i % len(default_colors)] for i, sid in enumerate(sample_ids)}
        
        vis_dir = f"{self.save_dir}/alignment_output/"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        ref_id = self.reference_sample_id
        ref_data = self.read_aligned_data(ref_id)  
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        ref_extent = None
        if isinstance(ref_data, np.ndarray):
            if len(ref_data.shape) == 2:  
                ax.imshow(ref_data, cmap='gray', alpha=alpha_img, zorder=1)
            else:  
                ax.imshow(ref_data, alpha=alpha_img, zorder=1)

            ref_height, ref_width = ref_data.shape[:2]
            ref_extent = (0, ref_width, 0, ref_height)

            ax.set_xlim(0, ref_width)
            ax.set_ylim(0, ref_height)
        
        elif isinstance(ref_data, sc.AnnData):
            ref_coords = ref_data.obsm['aligned_spatial']
            ax.scatter(ref_coords[:, 0], ref_coords[:, 1], 
                    s=point_size, color='gray', alpha=alpha_adata*0.6, 
                    zorder=2, label=f'Reference: {ref_id}')

            ref_x_min, ref_y_min = ref_coords.min(axis=0)
            ref_x_max, ref_y_max = ref_coords.max(axis=0)
            ref_extent = (ref_x_min, ref_x_max, ref_y_min, ref_y_max)

            ax.set_xlim(ref_x_min, ref_x_max)
            ax.set_ylim(ref_y_min, ref_y_max)
        
        for sample_id in self.input_slice_dict.keys():
            if sample_id == ref_id:
                continue
            
            print(f"Overlaying aligned slice for sample: {sample_id}")
            data = self.read_aligned_data(sample_id)  
            
            if isinstance(data, np.ndarray):
                if ref_extent is None:
                    raise ValueError("Reference extent not defined - cannot align target image")
        
                if len(data.shape) == 2:  
                    ax.imshow(data, cmap='gray', alpha=alpha_img*0.7, extent=ref_extent, zorder=3)
                else:  # RGB/RGBA
                    ax.imshow(data, alpha=alpha_img*0.7, extent=ref_extent, zorder=3)
            
            elif isinstance(data, sc.AnnData):
                coords = data.obsm['aligned_spatial']
                ax.scatter(coords[:, 0], coords[:, 1], 
                        s=point_size, color=color_map[sample_id], alpha=alpha_adata, 
                        zorder=4, label=f'Sample: {sample_id}')
        
        ax.invert_yaxis()
        
        ax.set_title(f'Aligned Slices (Reference: {ref_id})', fontsize=12, pad=10)
        ax.set_xlabel('X Coordinate (aligned)', fontsize=10)
        ax.set_ylabel('Y Coordinate (aligned)', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.9)
        plt.subplots_adjust(right=0.85)
        
        plt.tight_layout()
        if save_fig:
            save_path = f"{vis_dir}/aligned_slices_visualization.png"
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Combined visualization saved to: {save_path}")
        
        plt.show()
        
        plt.close(fig)
        del fig, ax


    def SpaMOMA_translate(self,adata_target,adata_src,save_path=None,k_neighbors=5,overlap_distance_threshold=20):
        if save_path is None:
            raise ValueError("Parameter save_path cannot be empty!")
        
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  
            
        adata_translated=translator.translate(adata_target.copy(),adata_src.copy(),k_neighbors,overlap_distance_threshold)
        adata_translated.write(save_path)
        
        return adata_translated
    
    def SpaMOMA_clustering(self,adata1,modality1,adata2,modality2,data_type,n_domains,method_list,cluster_types,add_key='consensus_clusters'):
        
        if {modality1,modality2} !=  {'rna', 'peak'} and  {modality1,modality2}!= {'rna', 'protein'} and {modality1,modality2} != {'gene_activity_score', 'peak'} and {modality1,modality2} != {'gene_activity_score', 'protein'}:
            raise ValueError(f"Unsupported modality combination: {modality1} + {modality2}")

        
        adata_final=adata1.copy()
        if 'SpatialGlue' in method_list:
            from .downstream_function.clustering.run_SpatialGlue import run_SpatialGlue
            if data_type not in ['10x','Stereo-CITE-seq', 'Spatial-epigenome-transcriptome','SPOTS']:
                raise ValueError(f"Unsupported data type for SpatialGlue: {data_type}")
            
            adata_final.obsm['SpatialGlue_emb']=run_SpatialGlue(adata1.copy(),modality1,adata2.copy(),modality2,data_type)
        if 'MISO' in method_list:    
            from .downstream_function.clustering.run_MISO import run_MISO
            adata_final.obsm['MISO_emb']=run_MISO(adata1.copy(),modality1,adata2.copy(),modality2)
        if 'PRESENT' in method_list:
            from .downstream_function.clustering.run_PRESENT import run_PRESENT
            adata_PRESENT=run_PRESENT(adata1.copy(),modality1,adata2.copy(),modality2)
            
            adata_PRESENT = adata_PRESENT[adata_final.obs_names, :]
        
            adata_final.obsm['PRESENT_emb'] = adata_PRESENT.obsm['embeddings'].copy()
        
        
        from .downstream_function.clustering.cluster_utils import run_leiden,run_louvain,run_mclust
        cluster_cols=[]
        for method in method_list:
            if 'louvain' in cluster_types or 'leiden' in cluster_types:
                sc.pp.neighbors(adata_final,use_rep=f'{method}_emb')
                
            if 'leiden' in cluster_types:
                adata_final = run_leiden(adata_final, n_cluster=n_domains, use_rep=f'{method}_emb', key_added=f'{method}_leiden')
                cluster_cols.append(f'{method}_leiden')
                
            if 'louvain' in cluster_types:
                adata_final = run_louvain(adata_final, n_cluster=n_domains, use_rep=f'{method}_emb', key_added=f'{method}_louvain')
                cluster_cols.append(f'{method}_louvain')
            if 'mclust' in cluster_types:
                run_mclust(adata_final, key=f'{method}_emb', add_key=f'{method}_mclust', n_clusters=n_domains, use_pca=True)
                cluster_cols.append(f'{method}_mclust')
            
        
        
        for col in cluster_cols:
            adata_final.obs[col] = adata_final.obs[col].astype('float')
            nan_count = adata_final.obs[col].isna().sum()
            
            if nan_count > 0:
                adata_final.obs[col] = adata_final.obs[col].fillna(0)
            
            adata_final.obs[col] = adata_final.obs[col].astype(int)

        from .downstream_function.clustering.cluster_utils import consensus_vote_from_existing
        adata_final = consensus_vote_from_existing(
            adata=adata_final,
            cluster_cols=cluster_cols,
            n_clusters=n_domains,  
            add_key=add_key
        )
        for col in cluster_cols:
            adata_final.obs[col]=adata_final.obs[col].astype(str)
        
        return adata_final
    

        
    def SpaMOMA_clustering_multiple_slices(self,multiple_slices_dict,reference_modality_dict,batch_key='batch',n_domains=10,spatial_key='aligned_spatial',add_key='integrative_clusters',overlap_distance_threshold=20,
                                           smooth=True,k_smooth=6,smooth_outlier_threshold=0.7):
        from .downstream_function.clustering.run_PRESENT import run_PRESENT_multiple_slices
        from .downstream_function.clustering.cluster_utils import domain_label_transfer,run_leiden,identify_overlapping_regions_multi_slices,spatial_neighborhood_voting
        
        if len(list(multiple_slices_dict.keys()))>2:
            raise ValueError("Only two slices are supported for joint clustering.")
        
        multiple_slices_dict=identify_overlapping_regions_multi_slices(multiple_slices_dict,distance_threshold=overlap_distance_threshold,spatial_key=spatial_key)
        
        adata_PRESENT=run_PRESENT_multiple_slices(multiple_slices_dict,batch_key,spatial_key)
        sc.pp.neighbors(adata_PRESENT,use_rep=f'PRESENT_emb')
        adata_PRESENT=run_leiden(adata_PRESENT,n_cluster=n_domains,use_rep='PRESENT_emb',key_added=add_key)

        labeled_full_adata_dict = {}  

        for sample_id,transfer_reference_omics in reference_modality_dict.items():
            full_adata = multiple_slices_dict[sample_id][transfer_reference_omics].copy()
            full_adata_with_label = domain_label_transfer(
                batch_id=sample_id,
                full_adata=full_adata,
                joint_cluster_adata=adata_PRESENT,
                domain_key=add_key,
                seed=0
            )
            if smooth==True:
                full_adata_with_label = spatial_neighborhood_voting(
                    full_adata_with_label,
                    cluster_col=add_key,
                    spatial_key=spatial_key,
                    k=k_smooth,
                    outlier_threshold=smooth_outlier_threshold,
                    new_cluster_col=f'{add_key}_smoothed'
                )
            
            labeled_full_adata_dict[sample_id] = full_adata_with_label
            
        return labeled_full_adata_dict
    
    def SpaMOMA_enhance_resolution(self,img_path,adata,pseudo_spot_size=10,weight_decay=4,n_neighbors=4):
        from .downstream_function.resolution_enhancement.enhance_resolution import enhance_resolution
        enhanced_adata=enhance_resolution(adata,img_path,pseudo_spot_size,weight_decay,n_neighbors)
        return enhanced_adata