import cv2
from pathlib import Path
import easyslam


def load_sys_conf(sys_conf_path):
    def parse_node(node):
        result = {}
        if node.isNone():
            return None

        # If the node is a map (dictionary in YAML)
        if node.type() == cv2.FileNode_MAP:
            keys = node.keys()
            for key in keys:
                result[key] = parse_node(node.getNode(key))
        
        # If the node is a sequence (list in YAML)
        elif node.type() == cv2.FileNode_SEQ:
            result = []
            for i in range(node.size()):
                result.append(parse_node(node.at(i)))
        
        # If the node is a single value
        elif node.isInt():
            result = int(node.real())
        elif node.isReal():
            result = float(node.real())
        elif node.isString():
            result = str(node.string())
        
        return result
    
    fs = cv2.FileStorage(sys_conf_path, cv2.FILE_STORAGE_READ)
    sys_conf = parse_node(fs.root())
    fs.release()
    return sys_conf


def kitti_slam(sensor_type, conf_path, viewer=False):
    # sensor_type = 'stereo'
    sensor_type = sensor_type
    # sys_conf_path = Path('config/stereo_kitti.yaml')
    sys_conf = load_sys_conf(conf_path)
    local_feature_conf = sys_conf['LocalFeature']
    global_feature_conf = sys_conf['GlobalFeature']
    matcher_conf = sys_conf['Matcher']

    slam = easyslam.EASY_SLAM(sensor_type, sys_conf, local_feature_conf, global_feature_conf, matcher_conf)
    slam.initialize(sensor_type, sys_conf, local_feature_conf)
    slam.tracking(sensor_type, sys_conf, local_feature_conf)
    slam.mapping(sensor_type, sys_conf, local_feature_conf, global_feature_conf)