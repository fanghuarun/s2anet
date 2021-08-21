# model settings
_base_ = './retinanet_obb_r50_fpn_1x_dota.py'
model = dict(
    bbox_head=dict(
        type='FreeAnchorHeadRotated',
        num_classes=16,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_angles=[0., ],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))

