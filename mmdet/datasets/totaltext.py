import math
import os,re,math
import os.path as osp
import mmcv,json
import numpy as np

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



IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

IMAGE_PREFIX = "img"
IMAGE_TYPE = ".jpg"

ANN_PREFIX = "poly_gt_img"
ANN_TYPE = ".txt"

def acquire_id(file_name):
    result = re.match(r'poly_gt_img(\d+).txt',file_name)
    return result.group(1)

def get_annfile_name(id):
    return ANN_PREFIX+str(id)+ANN_TYPE

def get_imagefile_name(id):
    return IMAGE_PREFIX+str(id)+IMAGE_TYPE

def filter_image(*args,**kwargs):
    """
    filter image.return False will be ignored;
    :param args:
    :param kwargs:
    :return:
    """
    return True

def get_label(xys,orct,transpition):
    return 1

def consult_obbox(box:np.array):
    """
    逆时针为正方向
    :param box:
    :return:
    """
    x,y = box[0]
    w,h = box[1]
    angle = box[2]

    if w < h:
        angle = angle - 90
        w,h = h,w

    angle = angle /360.0 * 2 * math.pi
    return [x,y,w,h,angle]

def consultline(line:str)->tuple:
    """

    :param line:
    :return:( (x,y,w,h,a) ,label , is_ignore )
    """
    xs,ys,orct,transption = line.split(",")
    # xs = re.search(r'\[\[(.*)\]\]',xs).group(1).strip().split(" ")
    xs = re.findall(r'\d+',xs)
    ys = re.findall(r'\d+',ys)
    # ys =  re.search(r'\[\[(.*)\]\]',ys).group(1).strip().split(" ")
    orct =  re.search(r'\[(.*)\]',orct).group(1)
    transption =  re.search(r'\[(.*)\]',transption).group(1)

    xys = np.array([[int(x),int(y)] for x,y in zip(xs,ys)])
    gt_bbox = cv2.minAreaRect(xys)
    # gt_bbox = [
    #         gt_bbox[0][0], gt_bbox[0][1], gt_bbox[1][0], gt_bbox[1][1], gt_bbox[2] / 360.0 * 2 * math.pi
    #     ]
    gt_bbox = consult_obbox(gt_bbox)
    gt_label = get_label(xys,orct,transption)
    is_ignore = not filter_image()
    return gt_bbox,gt_label,is_ignore



def consult_annfile(ann_file_name,imagefile_name , image_width = IMAGE_WIDTH,image_height = IMAGE_HEIGHT):

    f = open(ann_file_name,encoding='UTF-8')
    gt_boxs = []
    gt_labels = []
    ignore_boxs = []
    ignore_labels = []
    try:

        for line in f.readlines():
            line = line.strip()
            xywha, label ,isignore = consultline(line)
            if isignore:
                ignore_boxs.append(xywha)
                ignore_labels.append(label)
            else:
                gt_boxs.append(xywha)
                gt_labels.append(label)

        if not gt_boxs:
            gt_boxs = np.zeros((0, 5))
            gt_labels = np.zeros((0,))
        else:
            gt_boxs = np.array(gt_boxs, ndmin=2)
            gt_labels = np.array(gt_labels)

        if not ignore_boxs:
            ignore_boxs = np.zeros((0, 5))
            ignore_labels = np.zeros((0,))
        else:
            ignore_boxs = np.array(ignore_boxs, ndmin=2)
            ignore_labels = np.array(ignore_labels)
        _ann = {
            'filename': imagefile_name,
            'width': image_width,
            'height': image_height,
            'ann': {
                'bboxes': gt_boxs,
                'labels': gt_labels,
                'bboxes_ignore': ignore_boxs,
                'labels_ignore': ignore_labels,
            },
        }
        return _ann



    except Exception as e:
        print("load ann_file data:",ann_file_name,"appear error")
        raise e
    finally:

        f.close()


def load_ann_file(ann_file):
    _ann_dir = ann_file
    assert osp.isdir(_ann_dir)
    file_names = os.listdir(_ann_dir)
    ids = [acquire_id(file_name) for file_name in file_names]
    _result = []
    for _index in ids:
        ann_file_name = get_annfile_name(_index)
        ann_file_name = osp.join(_ann_dir,ann_file_name)
        image_file_name = get_imagefile_name(_index)
        _result.append(consult_annfile(ann_file_name,image_file_name))
    return _result





@DATASETS.register_module
class TotalTextDataset(CustomDataset):
    CLASSES = ('text',)


    def __init__(self,*args,**kwargs):

        super(TotalTextDataset, self).__init__(*args,**kwargs)



    def load_annotations(self, ann_file):
        datasetroot =  osp.dirname(osp.dirname(osp.dirname(ann_file)))
        wh_info_json = osp.join(datasetroot,"img_size_cache.json")
        wh_info = json.loads(wh_info_json)

        _ann_dir = ann_file
        assert osp.isdir(_ann_dir)
        file_names = os.listdir(_ann_dir)
        ids = [acquire_id(file_name) for file_name in file_names]
        _result = []
        for _index in ids:
            ann_file_name = get_annfile_name(_index)
            ann_file_name = osp.join(_ann_dir, ann_file_name)
            image_file_name = get_imagefile_name(_index)
            image_file_name = osp.abspath(image_file_name)
            _result.append(consult_annfile(ann_file_name, image_file_name,
                                           image_width=wh_info[image_file_name].get("w",IMAGE_WIDTH),
                                           image_height=wh_info[image_file_name].get("h",IMAGE_HEIGHT)))
        return _result
