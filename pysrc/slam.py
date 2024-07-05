# import os
# import sys
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
from pathlib import Path
# import yaml
import cv2


class EASY_SLAM:
    def __init__(self, sensor_type, local_feature_conf, global_feature_conf, matcher_conf, sys_conf_path, dataset_path):
        self.sensor_type = sensor_type
        self.local_feature_conf = local_feature_conf
        self.global_feature_conf = global_feature_conf
        self.matcher_conf = matcher_conf
        # self.sys_conf_path = sys_conf_path
        # self.dataset_path = dataset_path

        self.load_sys_conf()
        self.load_dataset()
        self.init_local_feature()
        self.init_global_feature()
        self.init_matcher()
    

    # load slam dataset conf. same format as ORB_SLAM2.
    def load_sys_conf(self):
        # Load config
        # sys_conf_path = Path(self.sys_conf_path)
        # sys_conf = yaml.load(open(sys_conf_path, 'r'), Loader=yaml.FullLoader)
        fs = cv2.FileStorage(self.sys_conf_path, cv2.FILE_STORAGE_READ)
        # dataset_conf = yaml.load(open(dataset_path, 'r'), Loader=yaml.FullLoader)
        # self.sys_conf = sys_conf
        # self.dataset_conf = dataset_conf


#%%
from pathlib import Path
# import yaml
import cv2

conf_path = "/home/keunmo/workspace/easy_slam/Examples/KITTI00-02.yaml"
# conf = yaml.load(open(conf_path, 'r'), Loader=yaml.FullLoader)
fs = cv2.FileStorage(conf_path, cv2.FILE_STORAGE_READ)
# %%
# investigate the structure of the conf file

fs.getNode("Camera.fx").real()
# %%
