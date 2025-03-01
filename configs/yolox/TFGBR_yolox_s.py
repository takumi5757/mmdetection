_base_ = ["../_base_/schedules/schedule_1x.py", "../_base_/default_runtime.py"]
# h,w 32の倍数
img_scale = (960, 960)
# model settings
model = dict(
    type="YOLOX",
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
    ),
    bbox_head=dict(type="YOLOXHead", num_classes=1, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)
seed = 5757

# dataset settings
data_root = "/workspace"
dataset_type = "CocoDataset"
classes = ("cots",)
train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        ann_file="/workspace/annotations_train.json",
        img_prefix="/workspace/images",
        classes=classes,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
            ),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file="/workspace/20220115_having_annotations_valid.json",
        img_prefix="/workspace/images",
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file="/workspace/20220115_having_annotations_valid.json",
        img_prefix="/workspace/images",
        classes=classes,
        pipeline=test_pipeline,
    ),
)

# optimizer
# default 8 gpu 0.01
#    lr=0.01/8だとevaluation mAPが０に張り付いたまま（70epまで確認）
# experiment
optimizer = dict(
    type="SGD",
    lr=0.01 / 4,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=None)

max_epochs = 80
num_last_epochs = 15
resume_from = "/workspace/mmdetection/work_dirs/TFGBR_yolox_s/epoch_60.pth"
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy="YOLOX",
    warmup="exp",
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05,
)

runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(
        type="SyncNormHook",
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48,
    ),
    dict(
        type="ExpMomentumEMAHook", resume_from=resume_from, momentum=0.0001, priority=49
    ),
]

# If the performance of model is pool, the `eval_res` may be an
# empty dict and it will raise exception when `self.save_best` is
# not None. More details at
# https://github.com/open-mmlab/mmdetection/issues/6265.

checkpoint_config = dict(interval=interval)
evaluation = dict(
    #     save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric="bbox",
)
log_config = dict(interval=50)
