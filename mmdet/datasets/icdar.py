import math
import os,re,math
import os.path as osp
import mmcv
import numpy as np

# from DOTA_devkit.ResultMerge_multi_process import mergebypoly
# from DOTA_devkit.dota_evaluation_task1 import voc_eval
# from mmdet.core import rotated_box_to_poly_single
from .custom import CustomDataset
from .registry import DATASETS


TEST_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))
TEST_DATA_ROOT = osp.join(TEST_ROOT,"data")
TEST_TRAIN_ANN = osp.join(TEST_DATA_ROOT,"ICDAR15-Train","ann")
TEST_TRAIN_IMG = osp.join(TEST_DATA_ROOT,"ICDAR15-Train","image")

IMAGE_PREFIX = "img_"
IMAGE_TYPE = ".jpg"
ANN_FILE_PREFIX = "gt_img_"
ANN_FILE_TYPE = ".txt"
import cv2


def acquire_img_idx(ann_dir):
    assert osp.isdir(ann_dir), "ann_file must is dir."
    file_names = os.listdir(ann_dir)
    idxs = [int(re.search(r'gt_img_(\d+)\.txt', file_name).group(1)) for file_name in file_names]
    return idxs

def consult_line(line:str,cat2label):
    try:
        line = line.strip()
        if line.startswith('\ufeff'):
            line = line[1:]
        x1, y1, x2, y2, x3, y3, x4, y4 = line.split(",")[:8]
        gt_box = cv2.minAreaRect(
            np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]]))
        gt_box = np.array([
            gt_box[0][0], gt_box[0][1], gt_box[1][0], gt_box[1][1], gt_box[2] / 360.0 * 2 * math.pi
        ])
        gt_label = cat2label["text"]
        return gt_box, gt_label
    except Exception as e:
        print("----------------------------------------------------")
        print("the process of consulting line",line," appear wrong")
        print("----------------------------------------------------")

        raise e

def filter_image(box)->bool:

    return True



def acquire_img_obbs_labels(ann_file, cat2label):
    f = open(ann_file,encoding="utf-8")
    gts_str = f.readlines()
    gt_boxs = []
    gt_labels = []
    boxs_ignore = []
    labels_ignore = []
    for line in gts_str:
        gt_box,gt_label = consult_line(line,cat2label)
        if not filter_image(gt_box):

            gt_boxs.append(gt_box)
            gt_labels.append(gt_label)
        else:
            boxs_ignore.append(gt_box)
            labels_ignore.append(gt_label)

    if not gt_boxs:
        gt_boxs = np.zeros((0, 5))
        gt_labels = np.zeros((0,))
    else:
        gt_boxs = np.array(gt_boxs,ndmin=2)
        gt_labels = np.array(gt_labels)

    if not boxs_ignore:
        boxs_ignore = np.zeros((0, 5))
        labels_ignore = np.zeros((0,))
    else:
        boxs_ignore = np.array(boxs_ignore, ndmin=2)
        labels_ignore = np.array(labels_ignore)


    f.close()
    return gt_boxs, gt_labels,boxs_ignore,labels_ignore


@DATASETS.register_module
class ICDAR15Dataset(CustomDataset):
    CLASSES = ('text',)


    def __init__(self,*args,**kwargs):

        super(ICDAR15Dataset, self).__init__(*args,**kwargs)



    def load_annotations(self, ann_file):
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}

        _ann_dir = ann_file
        idxs = acquire_img_idx(_ann_dir)

        result = []
        for idx in idxs:
            image_name = IMAGE_PREFIX + str(idx) + IMAGE_TYPE
            annfile_name = osp.join(_ann_dir, ANN_FILE_PREFIX + str(idx) + ANN_FILE_TYPE)
            gt_boxs, gt_labels,box_ignore,label_ignore = acquire_img_obbs_labels(annfile_name, self.cat2label)
            _ann = {
                'filename': image_name,
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': gt_boxs,
                    'labels': gt_labels,
                    'bboxes_ignore': box_ignore,
                    'labels_ignore':label_ignore,
                },
            }
            result.append(_ann)
        return result

