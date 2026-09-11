
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import sys
import math
from .utils.visualization import *
import torchvision.transforms as T
from .utils.train_roma_outdoor import get_model
import random
from typing import List, Dict, Optional, Tuple, Union
from .utils.robust_loss_mod import RobustLosses

import torch.nn as nn  
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class AugmentedWarpDataset(Dataset):
    def __init__(self, img_paths: List[str], 
                 epoch_angle_config: Dict[Tuple[int, int], List[float]],
                 angle_jitter: float = 5.0,
                 flip_prob: float = 0.3,
                 use_color_aug: bool = True,
                 colorjitter_params: Dict = None,
                 use_channel_shuffle: bool = False,
                 resize: Tuple[int, int] = (560, 560),
                 normalize_params: Dict = None,
                 transform: Optional[T.Compose] = None):

        self.img_paths = self._filter_valid_paths(img_paths)
        self.epoch_angle_config = epoch_angle_config
        self.angle_jitter = angle_jitter
        self.flip_prob = flip_prob

        self.use_color_aug = use_color_aug
        self.use_channel_shuffle = use_channel_shuffle
        self.colorjitter_params = colorjitter_params or {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.8, 1.2),
            'hue': (-0.05, 0.05)
        }
        self.colorjitter = T.ColorJitter(**self.colorjitter_params) if use_color_aug else None
        
        self.current_base_angles = list(epoch_angle_config.values())[0]
        self.current_num_base_angles = len(self.current_base_angles)
        
        self.resize = resize
        self.H, self.W = resize
        
        normalize_default = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.normalize_params = normalize_params or normalize_default
        
        self.transform = transform if transform is not None else T.Compose([
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean=self.normalize_params['mean'], std=self.normalize_params['std']),
        ])

    def set_current_epoch(self, epoch: int):
        for (start_epoch, end_epoch), base_angles in self.epoch_angle_config.items():
            if start_epoch <= epoch <= end_epoch:
                self.current_base_angles = base_angles
                self.current_num_base_angles = len(base_angles)
                print(f"Epoch {epoch}: Switched base angle set to {base_angles} (total {self.current_num_base_angles} angles)")
                break

    def set_angle_range(self, new_range: Tuple[float, float]):
        pass

    def _filter_valid_paths(self, img_paths: List[str]) -> List[str]:
        valid_paths = []
        for path in img_paths:
            if os.path.exists(path) and self._is_image_file(path):
                valid_paths.append(path)
            else:
                print(f"Warning: Skipping invalid path/non-image file {path}")
        if not valid_paths:
            raise ValueError("No valid image paths found, please check the img_paths parameter")
        return valid_paths

    @staticmethod
    def _is_image_file(path: str) -> bool:
        img_suffixes = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        return path.lower().endswith(img_suffixes)

    def _get_aug_params(self, base_angle: float) -> Tuple[Dict, Dict, str]:
        jitter = random.uniform(-self.angle_jitter, self.angle_jitter)
        final_angle = base_angle + jitter
        
        spatial_params = {
            'angle': final_angle,
            'flip_h': random.random() < self.flip_prob,
            'flip_v': random.random() < self.flip_prob * 0.5
        }
        
        color_params = {'type': 'none'} 
        color_ops = []
        if self.use_color_aug:
            if random.random() < 0.5:
                color_ops.append('colorjitter')
            if self.use_channel_shuffle and random.random() < 0.4:
                color_ops.append('channel_shuffle')
        
        if len(color_ops) > 0:
            color_params['type'] = '+'.join(color_ops)
        
        flag_parts = [
            f'base_{base_angle:.0f}_jitter_{jitter:.1f}_final_{final_angle:.1f}'
        ]
        if spatial_params['flip_h']:
            flag_parts.append('flip_h')
        if spatial_params['flip_v']:
            flag_parts.append('flip_v')
        if color_params['type'] != 'none':  
            flag_parts.append(color_params['type'])
        flag = '+'.join(flag_parts)
        
        return spatial_params, color_params, flag

    def _apply_spatial_aug(self, img: Image.Image, spatial_params: Dict) -> Tuple[Image.Image, Tuple[int, int]]:
        img_aug = img.copy()
        angle = spatial_params['angle']
        
        if spatial_params['flip_h']:
            img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
        if spatial_params['flip_v']:
            img_aug = img_aug.transpose(Image.FLIP_TOP_BOTTOM)
        
        if abs(angle) > 1e-3:
            w, h = img_aug.size
            center = (w / 2.0, h / 2.0)
            img_aug = img_aug.rotate(
                angle, 
                center=center,
                expand=True,
                resample=Image.BILINEAR
            )
        
        return img_aug, img_aug.size

    def _apply_color_aug(self, img: Image.Image, color_params: Dict) -> Image.Image:
        img_aug = img.copy()
        if color_params['type'] == 'none' or not self.use_color_aug:
            return img_aug
        
        if 'channel_shuffle' in color_params['type'] and self.use_channel_shuffle:
            channels = list(img_aug.split())
            random.shuffle(channels)
            img_aug = Image.merge('RGB', channels)
        
        if 'colorjitter' in color_params['type'] and self.colorjitter is not None:
            img_aug = self.colorjitter(img_aug)
        
        return img_aug

    def _compute_correspondence(self, wA: int, hA: int, spatial_params: Dict, wA_p: int, hA_p: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_coords = torch.linspace(0, self.W - 1, self.W, dtype=torch.float32)
        y_coords = torch.linspace(0, self.H - 1, self.H, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1)

        scale_x_A = wA / self.W
        scale_y_A = hA / self.H
        grid_orig_A = grid * torch.tensor([scale_x_A, scale_y_A], dtype=torch.float32)

        x_orig_A, y_orig_A = grid_orig_A[..., 0], grid_orig_A[..., 1]
        if spatial_params['flip_h']:
            x_orig_A = (wA - 1) - x_orig_A
        if spatial_params['flip_v']:
            y_orig_A = (hA - 1) - y_orig_A

        angle = spatial_params['angle']
        if abs(angle) < 1e-3:
            grid_orig_Ap = torch.stack([x_orig_A, y_orig_A], dim=-1)
        else:
            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            cx_A, cy_A = wA / 2.0, hA / 2.0
            cx_Ap, cy_Ap = wA_p / 2.0, hA_p / 2.0
            x_rot = cos_t * (x_orig_A - cx_A) + sin_t * (y_orig_A - cy_A) + cx_Ap
            y_rot = -sin_t * (x_orig_A - cx_A) + cos_t * (y_orig_A - cy_A) + cy_Ap
            grid_orig_Ap = torch.stack([x_rot, y_rot], dim=-1)

        scale_x_Ap = self.W / wA_p
        scale_y_Ap = self.H / hA_p
        gt_correspondence = grid_orig_Ap * torch.tensor([scale_x_Ap, scale_y_Ap], dtype=torch.float32)

        valid_x = (gt_correspondence[..., 0] >= 0) & (gt_correspondence[..., 0] < self.W)
        valid_y = (gt_correspondence[..., 1] >= 0) & (gt_correspondence[..., 1] < self.H)
        gt_valid_mask = (valid_x & valid_y).float()

        return gt_correspondence, gt_valid_mask

    def __len__(self) -> int:
        return len(self.img_paths) * self.current_num_base_angles

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, bool, int, float]]:
        try:
            img_idx = idx // self.current_num_base_angles
            angle_idx = idx % self.current_num_base_angles
            base_angle = self.current_base_angles[angle_idx]
            
            img_path = self.img_paths[img_idx]
            imgA = Image.open(img_path).convert('RGB')
            wA, hA = imgA.size

            spatial_params, color_params, flag = self._get_aug_params(base_angle)
            imgA_spatial, (wA_p, hA_p) = self._apply_spatial_aug(imgA, spatial_params)
            imgA_prime = self._apply_color_aug(imgA_spatial, color_params)

            im_A = self.transform(imgA)
            im_B = self.transform(imgA_prime)

            gt_correspondence, gt_valid_mask = self._compute_correspondence(
                wA, hA, spatial_params, wA_p, hA_p
            )

            return {
                'im_A': im_A,
                'im_B': im_B,
                'gt_correspondence': gt_correspondence,
                'gt_valid_mask': gt_valid_mask,
                'flag': flag,
                'img_path': img_path,
                'img_idx': img_idx,
                'base_angle': base_angle,
                'final_angle': spatial_params['angle'],
                'flip_h': spatial_params['flip_h'],
                'flip_v': spatial_params['flip_v'],
                'color_aug_type': color_params['type'],  
                'current_base_angles': self.current_base_angles
            }

        except Exception as e:
            img_idx_str = img_idx if 'img_idx' in locals() else 'unknown'
            base_angle_str = base_angle if 'base_angle' in locals() else 'unknown'
            print(f"Warning: Error processing index {idx} (original img index {img_idx_str}, base angle {base_angle_str}): {str(e)}, forcing return of sample 0")
            for _ in range(3):
                try:
                    return self.__getitem__(0)
                except Exception as e2:
                    print(f"Failed to retry getting sample 0: {str(e2)}")
                    continue
            
            raise RuntimeError(f"Failed to get valid samples (including sample 0), check image paths or augmentation logic") from e

def run_adaptation(img_paths,epochs=1000,batch_size=4,checkpoint_save_step=None,checkpoint_dir=None,
            epoch_angle_config=None,
             angle_jitter=10,flip_prob=0.5,use_color_aug=True,use_channel_shuffle=True,
             bn_layers_eval=True,
                gpu_id=3,log_flag=None
             ):
    
    
    if isinstance(epoch_angle_config, list):
        converted_config = {}
        for stage in epoch_angle_config:
            start = stage["start"]
            end = stage["end"]
            angles = stage["angles"]
            converted_config[(start, end)] = angles
        epoch_angle_config = converted_config
        
    if checkpoint_save_step==None:
        checkpoint_save_step=epochs

    os.makedirs(checkpoint_dir, exist_ok=True)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    print('training images:',img_paths)

    dataset = AugmentedWarpDataset(
        img_paths=img_paths,
        epoch_angle_config=epoch_angle_config,
        angle_jitter=angle_jitter,
        flip_prob=flip_prob,  
        use_color_aug=use_color_aug,  
        use_channel_shuffle=use_channel_shuffle, 
        colorjitter_params={  
            'brightness': (0.5, 1.5),
            'contrast': (0.5, 1.5),
            'saturation': (0.5, 1.15),
            'hue': (-0.1, 0.1)
        },
        resize=(560, 560)
    )
    

    model = get_model().to(device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "check_points", "roma_outdoor.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint,strict=True)
    model=model.to(device)
    
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for param in module.parameters():
                param.requires_grad = False
            if bn_layers_eval==True:
                module.eval()

    loss_fn = RobustLosses(
            ce_weight=0.01, 
            local_dist={1:4, 2:4, 4:8, 8:8},
            local_largest_scale=8,
            alpha = 0.5,
            c = 1e-4,)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    model.train()
    
    for epoch in range(epochs):

        dataset.set_current_epoch(epoch)  
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):

            batch['im_A'] = batch['im_A'].to(device)
            batch['im_B'] = batch['im_B'].to(device)
            batch['gt_correspondence'] = batch['gt_correspondence'].to(device)
            batch['gt_valid_mask'] = batch['gt_valid_mask'].to(device)
            batch['flag'] = batch['flag']  
            
            imA = batch['im_A']
            imB = batch['im_B']

            outputs = model({'im_A': imA, 'im_B': imB})
            loss = loss_fn(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        print(f"[{log_flag}] Epoch {epoch+1}/{epochs}, avg_loss={avg:.6f}")
        
        if (epoch + 1) % checkpoint_save_step == 0:
            save_path = os.path.join(checkpoint_dir, f'adapted_model_epoch_{epoch+1}.pth')
            torch.save( model.state_dict(),save_path)
            print(f'checkpoint saved to {save_path}')



