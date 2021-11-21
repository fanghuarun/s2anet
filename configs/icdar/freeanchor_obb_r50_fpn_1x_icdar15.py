# model settings
import os

model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
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
        loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1,
        iou_calculator=dict(type='BboxOverlaps2D_rotated')),
    bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.),
                    clip_border=True),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

# dataset settings
dataset_type = 'ICDAR15Dataset'

# data_root = 'data/HRSC2016/HRSC2016/'
data_root = os.path.join("data","ICDAR15-Train")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(100, 200), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=os.path.join(data_root,'ann'),
        img_prefix=os.path.join(data_root, 'image//'),
        pipeline=train_pipeline),
   )
# evaluation = dict(
#     gt_dir='data/HRSC2016/Test/Annotations/',
#     imagesetfile='data/HRSC2016/Test/test.txt')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
checkpoint_config = dict(interval=12)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
total_epochs = 36
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
