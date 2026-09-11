import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import random
import shutil
import sys
import time
import warnings
from PIL import Image
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# sys.path.append(str(Path(__file__).parents[1]))

sys.path.append('../utils/')
from ..utils import match_dense
from ..utils.viz import  display_keypoints, display_matches, fig2im, plot_images
from ..utils.get_matcher import Roma

warnings.simplefilter("ignore")

# @ref: https://docs.opencv.org/4.x/d0/d74/md__build_4_x-contrib_docs-lin64_opencv_doc_tutorials_calib3d_usac.html
# AND: https://opencv.org/blog/2021/06/09/evaluating-opencvs-new-ransacs
ransac_zoo = {
    "POSELIB": "LO-RANSAC",
    "CV2_RANSAC": cv2.RANSAC,
    "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
    "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
    "CV2_USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "CV2_USAC_PROSAC": cv2.USAC_PROSAC,
    "CV2_USAC_FAST": cv2.USAC_FAST,
    "CV2_USAC_ACCURATE": cv2.USAC_ACCURATE,
    "CV2_USAC_PARALLEL": cv2.USAC_PARALLEL,
}

DEVICE='cuda'


ROOT = Path(__file__).parent.parent
# some default values
DEFAULT_SETTING_THRESHOLD = 0.1
DEFAULT_SETTING_MAX_FEATURES = 2000
DEFAULT_DEFAULT_KEYPOINT_THRESHOLD = 0.0005
# DEFAULT_ENABLE_RANSAC = True
DEFAULT_ENABLE_RANSAC = False
DEFAULT_RANSAC_METHOD = "CV2_USAC_MAGSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_MAX_ITER = 2000
DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_MATCHING_THRESHOLD = 0.2
DEFAULT_SETTING_GEOMETRY = "Homography"
MATCHER_ZOO = None


def setup_matching_parameters():
    model_key='roma'
    
    match_threshold = 0.999  
    
    extract_max_keypoints = 4096
    # extract_max_keypoints = 10000

    

    force_resize = True
    image_height = 560  
    image_width = 560  
    
    
    choice_geometry_type = "Homography"  # or "fundamental" or "essential"/ "Homography"
    
    return {
        "model_key": model_key,
        "match_threshold": match_threshold,
        "extract_max_keypoints": extract_max_keypoints,
        "force_resize": force_resize,
        "image_height": image_height,
        "image_width": image_width,
        "choice_geometry_type": choice_geometry_type
    }




def run_matching(
        image0: np.ndarray,
        image1: np.ndarray,
        match_threshold: float,
        extract_max_keypoints: int,
        keypoint_threshold: float,
        key: str,
        ransac_method: str = DEFAULT_RANSAC_METHOD,
        ransac_reproj_threshold: int = DEFAULT_RANSAC_REPROJ_THRESHOLD,
        ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
        ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
        choice_geometry_type: str = DEFAULT_SETTING_GEOMETRY,
        matcher_zoo: Dict[str, Any] = None,
        force_resize: bool = False,
        image_width: int = 640,
        image_height: int = 480,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, float]],
    np.ndarray,
]:
    """Match two images using the given parameters.

    Args:
        image0 (np.ndarray): RGB image 0.
        image1 (np.ndarray): RGB image 1.
        match_threshold (float): match threshold.
        extract_max_keypoints (int): number of keypoints to extract.
        keypoint_threshold (float): keypoint threshold.
        key (str): key of the model to use.
        ransac_method (str, optional): RANSAC method to use.
        ransac_reproj_threshold (int, optional): RANSAC reprojection threshold.
        ransac_confidence (float, optional): RANSAC confidence level.
        ransac_max_iter (int, optional): RANSAC maximum number of iterations.
        choice_geometry_type (str, optional): setting of geometry estimation.
        matcher_zoo (Dict[str, Any], optional): matcher zoo. Defaults to None.
        force_resize (bool, optional): force resize. Defaults to False.
        image_width (int, optional): image width. Defaults to 640.
        image_height (int, optional): image height. Defaults to 480.
        use_cached_model (bool, optional): use cached model. Defaults to False.

    Returns:
        tuple:
            - output_keypoints (np.ndarray): image with keypoints.
            - output_matches_raw (np.ndarray): image with raw matches.
            - output_matches_ransac (np.ndarray): image with RANSAC matches.
            - num_matches (Dict[str, int]): number of raw and RANSAC matches.
            - configs (Dict[str, Dict[str, Any]]): match and feature extraction configs.
            - geom_info (Dict[str, Dict[str, float]]): geometry information.
            - output_wrapped (np.ndarray): wrapped images.
    """
    # image0 and image1 is RGB mode
    if image0 is None or image1 is None:
        print(
            "Error: No images found! Please upload two images or select an example."
        )
        raise gr.Error(
            "Error: No images found! Please upload two images or select an example."
        )
    # init output
    output_keypoints = None
    output_matches_raw = None
    output_matches_ransac = None

    # super slow!
    if "roma" in key.lower() and DEVICE == "cpu":
        gr.Info(
            f"Success! Please be patient and allow for about 2-3 minutes."
            f" Due to CPU inference, {key} is quiet slow."
        )
    t0 = time.time()
    model = matcher_zoo[key]
    match_conf = model["matcher"]
    # update match config
    match_conf["model"]["match_threshold"] = match_threshold
    match_conf["model"]["max_keypoints"] = extract_max_keypoints
    
    # matcher = get_model(match_conf)
    
    weights_path= match_conf['model']['weights_path']
    
    matcher= Roma(conf=match_conf['model'],weights_path=weights_path)
    print(f"current Roma instance ID: {id(matcher)}")  
    # print(f"Loading model using: {time.time() - t0:.3f}s")
    t1 = time.time()

    if model["dense"]:
        if not match_conf["preprocessing"].get("force_resize", False):
            match_conf["preprocessing"]["force_resize"] = force_resize
        else:
            print("preprocessing is already resized")
        if force_resize:
            match_conf["preprocessing"]["height"] = image_height
            match_conf["preprocessing"]["width"] = image_width
            print(f"Force resize to {image_width}x{image_height}")

        pred = match_dense.match_images(
            matcher, image0, image1, match_conf["preprocessing"], device=DEVICE
        )
        del matcher
        extract_conf = None
    

    # print(f"Matching images done using: {time.time() - t1:.3f}s")
    t1 = time.time()

    # plot images with keypoints
    titles = [
        "Image 0 - Keypoints",
        "Image 1 - Keypoints",
    ]
    output_keypoints = display_keypoints(pred, titles=titles)

    # plot images with raw matches
    titles = [
        "Image 0 - Raw matched keypoints",
        "Image 1 - Raw matched keypoints",
    ]
    output_matches_raw, num_matches_raw = display_matches(pred, titles=titles)

    t1 = time.time()

    # plot images with ransac matches
    titles = [
        "Image 0 - Ransac matched keypoints",
        "Image 1 - Ransac matched keypoints",
    ]
    # print(f"Display matches done using: {time.time() - t1:.3f}s")

    t1 = time.time()
    # plot wrapped images
    output_wrapped, warped_image = generate_warp_images(
        pred["image0_orig"],
        pred["image1_orig"],
        pred,
        choice_geometry_type,
    )
    plt.close("all")


    state_cache = pred
    state_cache["num_matches_raw"] = num_matches_raw
    state_cache["wrapped_image"] = warped_image


    tmp_state_cache = "output.pkl"

    return {
        
        "output_keypoints":output_keypoints,
        "output_matches_raw":output_matches_raw,
        "matches":{
            "num_raw_matches": num_matches_raw,
        },
        "conf":{
            "match_conf": match_conf,
            "extractor_conf": extract_conf,
        },

        "geom_info": pred.get("geom_info", {}),
    
        "output_wrapped":output_wrapped,
        "state_cache":state_cache,
        "tmp_state_cache":tmp_state_cache,
    
    }



def filter_matches(
        pred: Dict[str, Any],
        ransac_method: str = DEFAULT_RANSAC_METHOD,
        ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
        ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
        ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
        ransac_estimator: str = None,
):
    """
    Filter matches using RANSAC. If keypoints are available, filter by keypoints.
    If lines are available, filter by lines. If both keypoints and lines are
    available, filter by keypoints.

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, Any]: filtered matches.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None
    feature_type: Optional[str] = None
    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
        feature_type = "KEYPOINT"
    elif (
            "line_keypoints0_orig" in pred.keys()
            and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]
        feature_type = "LINE"
    else:
        return set_null_pred(feature_type, pred)
    if mkpts0 is None or mkpts0 is None:
        return set_null_pred(feature_type, pred)
    if ransac_method not in ransac_zoo.keys():
        ransac_method = DEFAULT_RANSAC_METHOD

    if len(mkpts0) < DEFAULT_MIN_NUM_MATCHES:
        return set_null_pred(feature_type, pred)

    geom_info = compute_geometry(
        pred,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
    )

    if "Homography" in geom_info.keys():
        mask = geom_info["mask_h"]
        if feature_type == "KEYPOINT":
            pred["mmkeypoints0_orig"] = mkpts0[mask]
            pred["mmkeypoints1_orig"] = mkpts1[mask]
            pred["mmconf"] = pred["mconf"][mask]
        elif feature_type == "LINE":
            pred["mline_keypoints0_orig"] = mkpts0[mask]
            pred["mline_keypoints1_orig"] = mkpts1[mask]
        pred["H"] = np.array(geom_info["Homography"])
    else:
        set_null_pred(feature_type, pred)
    # do not show mask
    geom_info.pop("mask_h", None)
    geom_info.pop("mask_f", None)
    pred["geom_info"] = geom_info
    return pred



def proc_ransac_matches(
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        ransac_method: str = DEFAULT_RANSAC_METHOD,
        ransac_reproj_threshold: float = 3.0,
        ransac_confidence: float = 0.99,
        ransac_max_iter: int = 2000,
        geometry_type: str = "Homography",
):
    if ransac_method.startswith("CV2"):
        print(
            f"ransac_method: {ransac_method}, geometry_type: {geometry_type}"
        )
        return _filter_matches_opencv(
            mkpts0,
            mkpts1,
            ransac_zoo[ransac_method],
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    elif ransac_method.startswith("POSELIB"):
        print(
            f"ransac_method: {ransac_method}, geometry_type: {geometry_type}"
        )
        return _filter_matches_poselib(
            mkpts0,
            mkpts1,
            None,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    else:
        raise NotImplementedError



def _filter_matches_opencv(
        kp0: np.ndarray,
        kp1: np.ndarray,
        method: int = cv2.RANSAC,
        reproj_threshold: float = 3.0,
        confidence: float = 0.99,
        max_iter: int = 2000,
        geometry_type: str = "Homography",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters matches between two sets of keypoints using OpenCV's findHomography.

    Args:
        kp0 (np.ndarray): Array of keypoints from the first image.
        kp1 (np.ndarray): Array of keypoints from the second image.
        method (int, optional): RANSAC method. Defaults to "cv2.RANSAC".
        reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to 3.0.
        confidence (float, optional): RANSAC confidence. Defaults to 0.99.
        max_iter (int, optional): RANSAC maximum iterations. Defaults to 2000.
        geometry_type (str, optional): Type of geometry. Defaults to "Homography".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Homography matrix and mask.
    """
    if geometry_type == "Homography":
        M, mask = cv2.findHomography(
            kp0,
            kp1,
            method=method,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
            maxIters=max_iter,
        )
    elif geometry_type == "Fundamental":
        M, mask = cv2.findFundamentalMat(
            kp0,
            kp1,
            method=method,
            ransacReprojThreshold=reproj_threshold,
            confidence=confidence,
            maxIters=max_iter,
        )
    mask = np.array(mask.ravel().astype("bool"), dtype="bool")
    return M, mask


def compute_geometry(
        pred: Dict[str, Any],
        ransac_method: str = DEFAULT_RANSAC_METHOD,
        ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
        ransac_confidence: float = DEFAULT_RANSAC_CONFIDENCE,
        ransac_max_iter: int = DEFAULT_RANSAC_MAX_ITER,
) -> Dict[str, List[float]]:
    """
    Compute geometric information of matches, including Fundamental matrix,
    Homography matrix, and rectification matrices (if available).

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, List[float]]: geometric information in form of a dict.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None

    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
    elif (
            "line_keypoints0_orig" in pred.keys()
            and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]

    if mkpts0 is not None and mkpts1 is not None:
        if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
            return {}
        geo_info: Dict[str, List[float]] = {}

        F, mask_f = proc_ransac_matches(
            mkpts0,
            mkpts1,
            ransac_method,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type="Fundamental",
        )

        if F is not None:
            geo_info["Fundamental"] = F.tolist()
            geo_info["mask_f"] = mask_f
        H, mask_h = proc_ransac_matches(
            mkpts1,
            mkpts0,
            ransac_method,
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type="Homography",
        )

        h0, w0, _ = pred["image0_orig"].shape
        if H is not None:
            geo_info["Homography"] = H.tolist()
            geo_info["mask_h"] = mask_h
            try:
                _, H1, H2 = cv2.stereoRectifyUncalibrated(
                    mkpts0.reshape(-1, 2),
                    mkpts1.reshape(-1, 2),
                    F,
                    imgSize=(w0, h0),
                )
                geo_info["H1"] = H1.tolist()
                geo_info["H2"] = H2.tolist()
            except cv2.error as e:
                print(
                    f"StereoRectifyUncalibrated failed, skip! error: {e}"
                )
        return geo_info
    else:
        return {}





def generate_warp_images(
        input_image0: np.ndarray,
        input_image1: np.ndarray,
        matches_info: Dict[str, Any],
        choice: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Changes the estimate of the geometric transformation used to align the images.

    Args:
        input_image0: First input image.
        input_image1: Second input image.
        matches_info: Dictionary containing information about the matches.
        choice: Type of geometric transformation to use ('Homography' or 'Fundamental') or 'No' to disable.

    Returns:
        A tuple containing the updated images and the warpped images.
    """
    if (
            matches_info is None
            or len(matches_info) < 1
            or "geom_info" not in matches_info.keys()
    ):
        return None, None
    geom_info = matches_info["geom_info"]
    warped_image = None
    if choice != "No":
        wrapped_image_pair, warped_image = wrap_images(
            input_image0, input_image1, geom_info, choice
        )
        return wrapped_image_pair, warped_image
    else:
        return None, None



def wrap_images(
        img0: np.ndarray,
        img1: np.ndarray,
        geo_info: Optional[Dict[str, List[float]]],
        geom_type: str,
) -> Tuple[Optional[str], Optional[Dict[str, List[float]]]]:
    """
    Wraps the images based on the geometric transformation used to align them.

    Args:
        img0: numpy array representing the first image.
        img1: numpy array representing the second image.
        geo_info: dictionary containing the geometric transformation information.
        geom_type: type of geometric transformation used to align the images.

    Returns:
        A tuple containing a base64 encoded image string and a dictionary with the transformation matrix.
    """
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    if geo_info is not None and len(geo_info) != 0:
        rectified_image0 = img0
        rectified_image1 = None
        if "Homography" not in geo_info:
            print(f"{geom_type} does not exist, maybe too few matches")
            return None, None

        H = np.array(geo_info["Homography"])

        title: List[str] = []
        if geom_type == "Homography":
            rectified_image1 = cv2.warpPerspective(img1, H, (w0, h0))
            title = ["Image 0", "Image 1 - warped"]
        elif geom_type == "Fundamental":
            print('Check if geom_type in geo_info:', geom_type not in geo_info)
            if geom_type not in geo_info:
                print(f"{geom_type} not exist, maybe too less matches")
                return None, None
            else:
                H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
                rectified_image0 = cv2.warpPerspective(img0, H1, (w0, h0))
                rectified_image1 = cv2.warpPerspective(img1, H2, (w1, h1))
                title = ["Image 0 - warped", "Image 1 - warped"]
        else:
            print("Error: Unknown geometry type")
        fig = plot_images(
            [rectified_image0.squeeze(), rectified_image1.squeeze()],
            title,
            dpi=300,
        )
        return fig2im(fig), rectified_image1
    else:
        return None, None


def filter_background_texture_points(
    kpts0: np.ndarray,  
    kpts1: np.ndarray,  
    image0: np.ndarray,  
    image1_aligned: np.ndarray,  
    mean_thresh: float = 0.05,  
    var_thresh: float = 1e-4,   
    patch_size: int = 5        
) -> np.ndarray:

    def get_local_stats(kpts: np.ndarray, img: np.ndarray, patch_size: int) -> tuple[np.ndarray, np.ndarray]:

        h, w = img.shape[:2]
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        img_gray = img_gray.astype(np.float32) / 255.0
        
        half_patch = patch_size // 2
        means = []
        vars = []
        
        for (x, y) in kpts:

            x_int = int(round(x))
            y_int = int(round(y))
            
            x1 = max(0, x_int - half_patch)
            x2 = min(w, x_int + half_patch + 1)
            y1 = max(0, y_int - half_patch)
            y2 = min(h, y_int + half_patch + 1)
            

            patch = img_gray[y1:y2, x1:x2]
            if patch.size == 0:
                means.append(0.0)
                vars.append(0.0)
                continue
            
            patch_mean = np.mean(patch)
            patch_var = np.var(patch)
            means.append(patch_mean)
            vars.append(patch_var)
        
        return np.array(means), np.array(vars)
    
    mean0, var0 = get_local_stats(kpts0, image0, patch_size)
    mean1, var1 = get_local_stats(kpts1, image1_aligned, patch_size)
    
    mask_non_black0 = mean0 > mean_thresh
    mask_non_black1 = mean1 > mean_thresh
    mask_texture0 = var0 > var_thresh
    mask_texture1 = var1 > var_thresh
    
    valid_mask = mask_non_black0 & mask_non_black1 & mask_texture0 & mask_texture1
    
    total = len(kpts0)
    valid = np.sum(valid_mask)

    # print(f"Background/texture filtering: {total} match points → {valid} retained")
    # print(f"Filtering reason statistics:")
    # print(f" - image0 all black: {total - np.sum(mask_non_black0)} points")
    # print(f" - image1_aligned all black: {total - np.sum(mask_non_black1)} points")
    # print(f" - image0 flat texture: {total - np.sum(mask_texture0)} points")
    # print(f" - image1_aligned flat texture: {total - np.sum(mask_texture1)} points")
    
    return valid_mask