# import os
# import sys
import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
from pathlib import Path
# import yaml
import cv2


class EASY_SLAM:
    def __init__(self, sensor_type, sys_conf, local_feature_conf, global_feature_conf, matcher_conf):
        self.sensor_type = sensor_type
        self.sys_conf = sys_conf
        self.local_feature_conf = local_feature_conf
        self.global_feature_conf = global_feature_conf
        if self.global_feature_conf == 'DBow3':
            self.voc = self.load_voc()
        self.matcher_conf = matcher_conf

    def initialize(self, sensor_type, sys_conf, local_feature_conf):
        # TODO
        pass

    def tracking(self, sensor_type, sys_conf, local_feature_conf):
        # TODO
        pass

    def mapping(self, sensor_type, sys_conf, local_feature_conf, global_feature_conf):
        # TODO
        pass

    