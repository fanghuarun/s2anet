import math

import torch
from torch import Tensor

K = 1


def kernel_func(d,s,k=K):
    return 1-1/(1+math.exp(-k*(d-s)))

def rotate_transformation(x,y,theta)->(float,float):
    """
    the linner transformation for coordinate.
    e1 = (1,0),e2 = (0,1)
    e1` = (cos\theta,sin\theta)
    e2` = (-sin\theta,cos\theta)
    (e1`,e2`) = (e1,e2)|cos\theta  -sin\theta|
                       |sin\theta   cos\theta|
    (e1',e2')|x| = (e1,e2)|cos\theta  -sin\theta| |x|
             |y|          |sin\theta   cos\theta| |y|

    :param x:
    :param y:
    :param theta: the \theta is about to the rotated angle.
    :return:
    the cordination after tansform.
    """
    x_ = math.cos(theta)*x-math.sin(theta)*y
    y_ = math.sin(theta)*x+math.cos(theta)*y
    return x_,y_

def judge_relative_loc(x,y,bbox):
    """
    judge the relative location (inside or outside) betweem P (x,y) and (c_x,c_y,w,h,theta).

    :param x:
    :param y:
    :param bbox(Tensor): bbox have shape(5) in <x,y,w,h,theta>.
    :return:It's close to 1 when P (x,y) in the area otherwise it ~ 0.
    """
    c_x, c_y, theta = bbox[0].item(), bbox[1].item(), bbox[4].item()
    w, h = bbox[2].item(), bbox[3].item()

    d_xy = math.sqrt(math.pow(x - c_x, 2) + math.pow(y-c_y, 2))

    if d_xy <=0 :

        d_w = 0
        d_h = 0
        return kernel_func(d_w, w) * kernel_func(d_h, h)

    if c_y - y >= 0:
        beta = theta + math.acos((c_x - x) / d_xy)
    else:
        beta = theta - math.acos((c_x - x) / d_xy)

    d_w = abs(d_xy * math.cos(beta))
    d_h = abs(d_xy * math.sin(beta))

    return kernel_func(d_w,w) * kernel_func(d_h,h)


def smallest_horizontal_bounding_box(bbox_1:Tensor,bbox_2:Tensor)->(int,int,int,int):
    """
    the smallest horizontal bounding box that covers bbox_1 and bbox_2.

    :param bbox_1:
    :param bbox_2:
    :return: (x_min,y_min,x_max,y_max)
    """
    c1_x, c1_y,c1_theta = bbox_1[0].item(), bbox_1[1].item(),bbox_1[4].item()
    # c1_x = 0 if c1_x<0 else c1_x
    # c1_y = 0 if c1_y<0 else c1_y

    c1_w_ = bbox_1[2].item()/ 2
    c1_h_ = bbox_1[3].item()/ 2

    xs = []
    ys = []
    
    x,y = rotate_transformation(c1_x - c1_w_,c1_y - c1_h_,c1_theta)
    xs.append(x)
    ys.append(y)

    x,y = rotate_transformation( c1_x + c1_w_,c1_y + c1_h_,c1_theta)
    xs.append(x)
    ys.append(y)

    c2_x, c2_y ,c2_theta = bbox_2[0].item(), bbox_2[1].item(),bbox_2[4].item()
    c2_w_ = bbox_2[2].item()*math.cos(c1_theta)/ 2
    c2_h_ = bbox_2[3].item()*math.sin(c1_theta)/ 2
    # c2_x = 0 if c2_x<0 else c2_x
    # c2_y = 0 if c2_y<0 else c2_y

    x,y = rotate_transformation( c2_x + c2_w_,c2_y + c2_h_,c2_theta)
    xs.append(x)
    ys.append(y)
    
    x,y = rotate_transformation(c2_x - c2_w_,c2_y - c2_h_,c2_theta)
    xs.append(x)
    ys.append(y)

    return min(xs),min(ys),max(xs),max(ys)

def PIou_betweem_box_and_box(bbox_1,bbox_2):
    x_min, y_min, x_max, y_max = smallest_horizontal_bounding_box(bbox_1, bbox_2)
    s_intersect = float(0)

    for x in range(int(x_min), int(x_max) + 1):
        for y in range(int(y_min), int(y_max) + 1):
            s_intersect = s_intersect + judge_relative_loc(x, y, bbox_1) * judge_relative_loc(x, y, bbox_2)
    s_uniou = bbox_1[2].item() * bbox_1[3].item() + bbox_2[2].item() * bbox_2[3].item() - s_intersect  # the value of union area.

    return s_intersect/s_uniou

def PIoU(bboxs_1:Tensor,bboxs_2:Tensor):
    """
    compute the Pixels-IoU (P IoU) betweem bboxs_1 and bboxs_2.
    :param bboxs_1:bboxes have shape (m, 5) in <x, y, w, h, a> format,
    :param bboxs_2:bboxes have shape (m, 5) in <x, y, w, h, a> format,
    :return:<Tensor> it record their Pixels-IoU (P IoU) which have the shape <m,>.

    """
    pious = []
    for bbox_1 in bboxs_1:
        _pious = []

        for bbox_2 in bboxs_2:
            x_min, y_min, x_max, y_max = smallest_horizontal_bounding_box(bbox_1, bbox_2)
            # the value of intersection area
            s_intersect = float(0)
            print("({},{},{},{})".format(x_min,y_min,x_max,y_max))
            for x in range(int(x_min), int(x_max) + 1):
                for y in range(int(y_min), int(y_max) + 1):
                    s_intersect = s_intersect + judge_relative_loc(x, y, bbox_1) * judge_relative_loc(x, y, bbox_2)
            s_uniou = bbox_1[2].item() * bbox_1[3].item() + bbox_2[2].item() * bbox_2[3].item() - s_intersect  # the value of union area.
            _pious.append(s_intersect / s_uniou)
        print(_pious)
        pious.append(_pious)
    return torch.tensor(pious)

    # for bboxs in zip(bboxs_1,bboxs_2):
    #     bbox_1 = bboxs[0]
    #     bbox_2 = bboxs[1]
    #     x_min, y_min, x_max, y_max = smallest_horizontal_bounding_box(bbox_1, bbox_2)
    #     # the value of intersection area
    #     s_intersect = float(0)
    #
    #     for x in range(int(x_min),int(x_max)+1):
    #         for y in range(int(y_min),int(y_max)+1):
    #             s_intersect = s_intersect+judge_relative_loc(x,y,bbox_1)*judge_relative_loc(x,y,bbox_2)
    #     s_uniou = bbox_1[2].item()*bbox_1[3].item()+bbox_2[2].item()*bbox_2[3].item()-s_intersect #the value of union area.
    #     pious.append(s_intersect/s_uniou)



if __name__ == '__main__':
    bboxs_1 = torch.tensor([
        [250,122,50,20,0.5],
        [122,41,40,60,0],

    ])
    bboxs_2 = torch.tensor([
        [1200, 52, 43, 75, 0.3],
        [262, 56, 42, 60, 0],
        [52, 100, 10, 5, 0.1]
    ])
    result = PIoU(bboxs_1,bboxs_2)
    print(result)
    print(result.shape)