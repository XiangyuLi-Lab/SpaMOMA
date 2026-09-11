from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=4.0,
        local_largest_scale=8,
        smooth_mask=False,
        depth_interpolation_mode="bilinear",  
        mask_depth_loss=False,
        relative_depth_error_threshold=0.05,
        alpha=1.,
        c=1e-3,
    ):
        super().__init__()
        self.robust = robust
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight  
        self.local_loss = local_loss 
        self.local_dist = local_dist  
        self.local_largest_scale = local_largest_scale  
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode  
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha 
        self.c = c  

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))  
   
            grid = torch.meshgrid(
                *[torch.linspace(-1 + 1/cls_res, 1 - 1/cls_res, steps=cls_res, device=device) 
                  for _ in range(2)], 
                indexing='ij'
            )
            grid = torch.stack((grid[1], grid[0]), dim=-1).reshape(C, 2)  
            
            gt_indices = (grid[None, :, None, None, :] - x2[:, None]).norm(dim=-1).min(dim=1).indices
        
    
        cls_loss = F.cross_entropy(scale_gm_cls, gt_indices, reduction='none')[prob > 0.99]
     
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:, 0], prob)
        
 
        if not torch.any(cls_loss):
            cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # print('gm_cls_loss',cls_loss,certainty_loss)
        return {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode="delta"):

        pred_flow = flow.permute(0, 2, 3, 1)
        epe = (pred_flow - x2).norm(dim=-1) 
  
        if scale == 1:
            valid_epe = epe[prob > 0.99]
            if len(valid_epe) > 0:
              
                pck_05 = (valid_epe < 0.5 * (2/512)).float().mean()
            
    
        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        
        valid_epe = epe[prob > 0.99]
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale  
        if len(valid_epe) > 0:
            reg_loss = cs**a * ((valid_epe / cs)** 2 + 1)**(a / 2)  
        else:
            reg_loss = torch.tensor(0.0, device=epe.device, requires_grad=True)
        
        # print('regression_loss',ce_loss,reg_loss)
        return {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }

    def forward(self, corresps, batch):
        """
        Forward pass: Calculate total loss across multiple scales
        corresps: Multi-scale correspondence dict from model output {scale: {predictions}}
        batch: Input batch data, including im_A, im_B, gt_correspondence, gt_valid_mask
        """
        scales = list(corresps.keys())
        tot_loss = 0.0
        scale_weights = {1:1.0, 2:1.0, 4:1.0, 8:1.0, 16:1.0}  
        # print('scale wights',scale_weights)
        prev_epe = None 
        
        for scale in scales:
            scale_corresps = corresps[scale]

            scale_certainty = scale_corresps["certainty"]
            flow_pre_delta = scale_corresps.get("flow_pre_delta")
            delta_cls = scale_corresps.get("delta_cls")
            offset_scale = scale_corresps.get("offset_scale")
            scale_gm_cls = scale_corresps.get("gm_cls")
            scale_gm_certainty = scale_corresps.get("gm_certainty")
            flow = scale_corresps["flow"]
            scale_gm_flow = scale_corresps.get("gm_flow")
            
    
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d") 
                b, h, w, _ = flow_pre_delta.shape
            else:
                b, _, h, w = scale_certainty.shape 
            

            gt_correspondence = batch["gt_correspondence"].clone()  
            gt_valid_mask = batch.get("gt_valid_mask", torch.ones_like(gt_correspondence[..., 0]))  
            

            H_gt, W_gt = gt_correspondence.shape[1], gt_correspondence.shape[2]
     
            gt_correspondence[..., 0] = (gt_correspondence[..., 0] / (W_gt - 1 + 1e-8)) * 2 - 1
  
            gt_correspondence[..., 1] = (gt_correspondence[..., 1] / (H_gt - 1 + 1e-8)) * 2 - 1
            
     
            gt_warp = F.interpolate(
                gt_correspondence.permute(0, 3, 1, 2), 
                size=(h, w),
                mode=self.depth_interpolation_mode,
                align_corners=False 
            ).permute(0, 2, 3, 1)  
            
      
            gt_prob = F.interpolate(
                gt_valid_mask.unsqueeze(1),  
                size=(h, w),
                mode="nearest-exact"
            ).squeeze(1)  # [B, h, w]
            gt_prob = torch.clamp(gt_prob, 0.0, 1.0) 
            
    
            if self.local_largest_scale >= scale and self.local_loss and prev_epe is not None:
                local_dist = self.local_dist[scale] if isinstance(self.local_dist, dict) else self.local_dist

                local_mask = (
                    F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                    < (2 / 512) * (local_dist * scale) 
                )
                gt_prob = gt_prob * local_mask.float()  
            
            if scale_gm_cls is not None:
                gm_losses = self.gm_cls_loss(gt_warp, gt_prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_losses[f"gm_certainty_loss_{scale}"] + gm_losses[f"gm_cls_loss_{scale}"]
                tot_loss += scale_weights[scale] * gm_loss
            
      
            delta_reg_losses = self.regression_loss(gt_warp, gt_prob, flow, scale_certainty, scale)
            delta_loss = self.ce_weight * delta_reg_losses[f"delta_certainty_loss_{scale}"] + delta_reg_losses[f"delta_regression_loss_{scale}"]
            tot_loss += scale_weights[scale] * delta_loss
           

            current_epe = (flow.permute(0, 2, 3, 1) - gt_warp).norm(dim=-1).detach()
            prev_epe = current_epe
        
 
        return tot_loss
    
    
    
    
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

def denormalize_image(tensor):
    """
    Denormalize image tensor (restore to 0-1 range, compatible with training Normalize)
    Input: tensor [C, H, W] (normalized with mean [0.485,0.456,0.406], std [0.229,0.224,0.225])
    Output: tensor [C, H, W] (range 0-1)
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def generate_warped_image(source_img, correspondence, img_size):
    """
    Generate warped image based on pixel correspondences (map source_img pixels to target positions)
    Input:
        source_img: Source image [C, H, W] (denormalized, 0-1 range)
        correspondence: Pixel correspondences [H, W, 2] (x,y format, target image coordinates)
        img_size: Target image size (H_target, W_target)
    Output:
        warped_img: Mapped image [C, H_target, W_target] (0-1 range)
    """
    C, H_src, W_src = source_img.shape
    H_target, W_target = img_size
    device = source_img.device

    warped_img = torch.zeros((C, H_target, W_target), device=device)
    
    x_target = correspondence[..., 0].clamp(0, W_target - 1).round().long()
    y_target = correspondence[..., 1].clamp(0, H_target - 1).round().long()
    
    y_src, x_src = torch.meshgrid(
        torch.arange(H_src, device=device), 
        torch.arange(W_src, device=device), 
        indexing='ij'
    )
    
    warped_img[:, y_target, x_target] = source_img[:, y_src, x_src]
    
    return warped_img

