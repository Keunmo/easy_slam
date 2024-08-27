import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import h5py
import numpy as np
import PIL.Image
import torch

from hloc import extractors, logger
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import list_h5_names
# from hloc.utils.parsers import parse_image_lists

"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""
confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "output": "feats-superpoint-n4096-rmax1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "resize_force": True,
        },
    },
    "superpoint_inloc": {
        "output": "feats-superpoint-n4096-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "r2d2": {
        "output": "feats-r2d2-n5000-r1024",
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "d2net-ss": {
        "output": "feats-d2net-ss",
        "model": {
            "name": "d2net",
            "multiscale": False,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "sosnet": {
        "output": "feats-sosnet",
        "model": {"name": "dog", "descriptor": "sosnet"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "disk": {
        "output": "feats-disk",
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    # Global descriptors
    "dir": {
        "output": "global-feats-dir",
        "model": {"name": "dir"},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    },
    "openibl": {
        "output": "global-feats-openibl",
        "model": {"name": "openibl"},
        "preprocessing": {"resize_max": 1024},
    },
    "eigenplaces": {
        "output": "global-feats-eigenplaces",
        "model": {"name": "eigenplaces"},
        "preprocessing": {"resize_max": 1024},
    },
}


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class FeatureExtractor:
    default_conf = {
        # "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
    }

    def __init__(self, conf: str, export_dir: Optional[Path] = None, 
                 as_half: bool = True):
        # self.conf = self.default_conf | confs[conf]
        self.conf = confs[conf]
        self.conf["preprocessing"] = self.default_conf | self.conf["preprocessing"]  # order matters
        # self.preproc_conf = self.default_conf | self.conf["preprocessing"]
        # self.conf = SimpleNamespace(**self.conf)
        self.export_dir = export_dir
        self.as_half = as_half
        # load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)
        # create export directory
        if self.export_dir is None:
            self.export_dir = Path(__file__).parent / "outputs"
            # self.export_dir.mkdir(exist_ok=True)
        self.feature_path = Path(self.export_dir, self.conf["output"] + ".h5")
        self.feature_path.parent.mkdir(exist_ok=True, parents=True)
        self.skip_names = set(
            list_h5_names(self.feature_path) if self.feature_path.exists() else ()
        )

    '''
    Extract features from an image. 
    image should be mat like, BGR, HWC format.
    '''
    @torch.no_grad()
    def extract(self, image, frameId: Optional[int] = None, save_features: bool = True):
        # preprocessing
        if self.conf["preprocessing"]["grayscale"] and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.conf["preprocessing"]["grayscale"] and len(image.shape) == 3:
            image = image[:, :, ::-1]  # BGR to RGB
        size = image.shape[:2][::-1] # original size. (W,H)
        if self.conf["preprocessing"]["resize_max"] and (
            self.conf["preprocessing"]["resize_force"] or max(size) > self.conf["preprocessing"]["resize_max"]
        ):
            scale = self.conf["preprocessing"]["resize_max"] / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf["preprocessing"]["interpolation"])

        if self.conf["preprocessing"]["grayscale"]:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0
        image = torch.from_numpy(image).float()[None]
        # preprocessing end

        pred = self.model({"image": image.to(self.device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = np.array(size)
        if "keypoints" in pred:
            size_new = np.array(image.shape[-2:][::-1])
            scales = (size / size_new).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(self.model, "detection_noise", 1) * scales.mean()

        if self.as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        
        if save_features:
            name = str(frameId) if frameId is not None else str(time.time())
            with h5py.File(str(self.feature_path), "a", libver="latest") as fd:
                try:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    for k, v in pred.items():
                        grp.create_dataset(k, data=v)
                    if "keypoints" in pred:
                        grp["keypoints"].attrs["uncertainty"] = uncertainty
                except OSError as error:
                    if "No space left on device" in error.args[0]:
                        logger.error(
                            "Out of disk space: storing features on disk can take "
                            "significant space, did you enable the as_half flag?"
                        )
                        del grp, fd[name]
                    raise error
        
        return pred


if __name__ == "__main__":
    # feature_extractor = FeatureExtractor("disk", export_dir=Path("../outputs/tests"))
    feature_extractor = FeatureExtractor("superpoint_aachen", export_dir=Path("../outputs/tests"))
    image = cv2.imread("/home/keunmo/workspace/Hierarchical-Localization/datasets/sacre_coeur/mapping/02928139_3448003521.jpg")
    pred = feature_extractor.extract(image, 0, True)
    print(pred.keys())
