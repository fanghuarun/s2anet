
import torch

def rbox2corners_torch(loc):
    cos_w = 0.5 * loc[:, 2:3] * torch.cos(loc[:, 4:5])
    sin_w = 0.5 * loc[:, 2:3] * torch.sin(loc[:, 4:5])
    cos_h = 0.5 * loc[:, 3:4] * torch.cos(loc[:, 4:5])
    sin_h = 0.5 * loc[:, 3:4] * torch.sin(loc[:, 4:5])
    x0 = loc[:, 0:1] + cos_w + sin_h
    y0 = loc[:, 1:2] - sin_w + cos_h
    x1 = loc[:, 0:1] - cos_w + sin_h
    y1 = loc[:, 1:2] + sin_w + cos_h
    x2 = loc[:, 0:1] - cos_w - sin_h
    y2 = loc[:, 1:2] + sin_w - cos_h
    x3 = loc[:, 0:1] + cos_w - sin_h
    y3 = loc[:, 1:2] - sin_w - cos_h
    rbox = torch.cat((x0, y0, x1, y1, x2, y2, x3, y3), -1)
    return rbox

def smallest_horizen_area(boxs1:torch.Tensor,boxs2:torch.Tensor)->torch.Tensor:
    """

    :param boxs1:<num,5>
    :param boxs2:<num,5>
    :return: <num,4> <x_min,x_max,y_min,y_max>.
    """
    rboxs1 = rbox2corners_torch(boxs1)
    rboxs2 = rbox2corners_torch(boxs2)
    rboxs = torch.cat((rboxs1,rboxs2),-1)
    x_area = rboxs[:,::2]
    y_area = rboxs[:,1::2]
    xmax_s,xmax_i = x_area.max(-1)
    xmin_s,xmin_i = x_area.min(-1)
    ymax_s,ymax_i = y_area.max(-1)
    ymin_s,ymin_i = y_area.min(-1)

    return torch.cat((xmin_s.unsqueeze(1),xmax_s.unsqueeze(1),ymin_s.unsqueeze(1),ymax_s.unsqueeze(1)),-1)

def grid_xy_in_area(x_min,x_max,y_min,y_max):

    xv, yv = torch.meshgrid(
        [torch.arange(x_min-10, x_max+10), torch.arange(y_min-10, y_max+10)])
    xy = torch.stack((xv, yv), -1)
    grid_xy = xy.reshape(-1, 2).float() + 0.5
    return grid_xy

def kernel_function(dis, k, t):
    # clamp to avoid nan
    factor = torch.clamp(-k * (dis - t), -50, 50)
    return 1.0 - 1.0 / (torch.exp(factor) + 1)

# loc --> num x dim x 5
# grid_xy --> num x dim x 2


def pixel_weights(loc:torch.Tensor, grid_xy:torch.Tensor, k):
    """

    :param loc: <
    :param grid_xy:
    :param k:
    :return:
    """
    xx = torch.pow(loc[:,0:2], 2).sum(1)
    yy = torch.pow(grid_xy, 2).sum(1)
    dis = xx + yy
    # dis - 2 * x * yT
    dis = dis-2*torch.mm(grid_xy,loc[0,0:2].reshape(2,1)).squeeze()
    # dis = torch.addmm(dis,loc[0,0:2],grid_xy.t(),alpha=-2,beta=1)
    dis = dis.clamp(min=1e-9).sqrt()  # for numerical stability

    a1 = loc[:, -1] - torch.acos((grid_xy[:, 0] - loc[:, 0]) / dis)
    a2 = loc[:, -1] + torch.acos((grid_xy[:, 0] - loc[:, 0]) / dis)
    a = torch.where(loc[:, 1] > grid_xy[:, 1], a1, a2)

    dis_w = dis * torch.abs(torch.cos(a))
    dis_h = dis * torch.abs(torch.sin(a))
    # return dis_h
    kernel_w = kernel_function(dis_w, k, loc[:, 2] / 2.)
    kernel_h = kernel_function(dis_h, k, loc[:, 3] / 2.)
    _pixel_weights = kernel_h*kernel_w

    return _pixel_weights



def HPIou_betweem_box_and_box(pred_box:torch.Tensor,gt_box:torch.Tensor,area,k):
    grid_xy = grid_xy_in_area(area[0],area[1],area[2],area[3])#<pm*pn,2>
    dim = grid_xy.size(0)
    _pred_boxs = pred_box.unsqueeze(0).expand(dim,5)
    _gt_boxs = gt_box.unsqueeze(0).expand(dim,5)

    area_p = pred_box[2] * pred_box[3]
    area_t = gt_box[2] * gt_box[3]

    pixel_p_weights = pixel_weights(_pred_boxs, grid_xy, k)
    pixel_t_weights = pixel_weights(_gt_boxs, grid_xy, k)

    inter_pixel_area = pixel_p_weights * pixel_t_weights
    intersection_area = torch.sum(inter_pixel_area, 0)

    union_area = area_p + area_t - intersection_area

    piou = intersection_area / (union_area + 1e-9)

    return piou



def HPIous(pred_boxs:torch.Tensor,gt_boxs:torch.Tensor,k):
    num = pred_boxs.size(0)

    _gt_boxs = gt_boxs.expand(num,5)
    areas = smallest_horizen_area(pred_boxs,_gt_boxs)
    pious = []
    for pred_box,gt_box,area in zip(pred_boxs,gt_boxs,areas):
        piou = HPIou_betweem_box_and_box(pred_box,gt_box,area,k)
        pious.append(piou.item())
    return torch.Tensor(pious)
