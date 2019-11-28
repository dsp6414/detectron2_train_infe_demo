import os
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA


# 数据集路径
DATASET_ROOT = '/home/watson/Documents/mask_THzDatasets'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')
TRAIN_JSON = os.path.join(ANN_ROOT, 'mask_THz_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'mask_THz_val.json')


# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "gun", "id": 1, "isthing": 1, "color": [220, 20, 60]},
    {"name": "phone", "id": 2, "isthing": 1, "color": [219, 142, 185]},
]


# 数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "iecas_THz_2019_train": (TRAIN_PATH, TRAIN_JSON),
    "iecas_THz_2019_val": (VAL_PATH, VAL_JSON),
}


def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("iecas_THz_2019_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "iecas_THz_2019_train"))
    MetadataCatalog.get("iecas_THz_2019_train").set(thing_classes=["gun", "phone"],
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
    DatasetCatalog.register("iecas_THz_2019_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "iecas_THz_2019_val"))
    MetadataCatalog.get("iecas_THz_2019_val").set(thing_classes=["gun", "phone"],
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)


# 查看数据集标注
def checkout_dataset_annotation(name="iecas_THz_2019_train"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, distributed=False, output_dir=output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    args.config_file = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("iecas_THz_2019_train",)
    cfg.DATASETS.TEST = ("iecas_THz_2019_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # 默认多线程为4; 用vscode调试时启用多线程会报错，要设为0
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 类别数
    cfg.MODEL.WEIGHTS = "/home/watson/Documents/pretrainedModel/Detectron2/R-50.pkl"    # 预训练模型权重
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size  
    ITERS_IN_ONE_EPOCH = int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg)

    # 注册数据集
    register_dataset()
    checkout_dataset_annotation(name="iecas_THz_2019_train")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
