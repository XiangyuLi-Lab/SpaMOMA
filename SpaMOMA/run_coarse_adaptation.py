# run_coarse.py

import os
import sys
import json
from .test_time_adaption import run_adaptation

if __name__ == "__main__":

    config = json.loads(sys.argv[1])
    
    run_adaptation(
        img_paths=config["img_paths"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        checkpoint_dir=config["checkpoint_dir"],
        epoch_angle_config=config["epoch_angle_config"],
        checkpoint_save_step=config['checkpoint_save_step'],
        angle_jitter=10,
        flip_prob=0.5,
        use_color_aug=True,
        use_channel_shuffle=True,
        bn_layers_eval=False,
        gpu_id=0,
        log_flag="coarse"
    )
    
