import sys
import torch
from PIL import Image
sys.path.append('../')
from romatch.models.model_zoo import roma_model
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

class Roma():

    def __init__(self,
                 weights_path=os.path.join(CURRENT_FILE_DIR, '../check_points/roma_outdoor.pth'),
                 conf=None
                 ):
        
        if weights_path is None:
            weights_path=os.path.join(CURRENT_FILE_DIR, '../check_points/roma_outdoor.pth')


        print(f"Loading Roma model: {weights_path}")
        weights = torch.load(weights_path, map_location="cpu")
        # load the model
        dinov2_weights=os.path.join(CURRENT_FILE_DIR, '../check_points/dinov2_vitl14_pretrain.pth')

        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        self.net = roma_model(
            resolution=(14 * 8 * 6, 14 * 8 * 6),
            upsample_preds=False,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
            # temp fix issue: https://github.com/Parskatt/RoMa/issues/26
            amp_dtype=torch.float32,
        )
        self.conf=conf

    def __call__(self, x):
        return self.forward(x)

    def forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        # Match
        warp, certainty = self.net.match(img0, img1, device=device)
        # Sample matches for estimation
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(
            matches, H_A, W_A, H_B, W_B
        )
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
            "corresps":{
                "warp":warp,
                "certainty":certainty
            }
            
     
        }

        return pred
    
    
   