
# loc --> num x dim x 5
# grid_xy --> num x dim x 2

import torch


def kernel_function(dis, k, t):
    # clamp to avoid nan
    factor = torch.clamp(-k * (dis - t), -50, 50)
    return 1.0 - 1.0 / (torch.exp(factor) + 1)

# loc --> num x dim x 5
# grid_xy --> num x dim x 2

