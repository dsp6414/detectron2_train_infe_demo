import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "detections"

# inference
INPUT_IMG_PATH = '/home/watson/Documents/mask_THzDatasets/val'
OUTPUT_IMG_PATH = '/home/watson/Documents/out'

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


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("iecas_THz_2019_train",)
    cfg.DATASETS.TEST = ("iecas_THz_2019_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # 默认 多线程为4; vscode调试时启用多线程会报错，要设为0
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 类别数
    # cfg.MODEL.WEIGHTS = "/home/watson/Documents/pretrainedModel/Detectron2/R-50.pkl"  # 预训练模型权重
    cfg.MODEL.WEIGHTS = '/home/watson/anaconda3/envs/pytorch13/detectron2/output/model_final.pth'   # 最终权重
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iteration = 1434/batch_size = 717 iters in one epoch
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
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    # 注册数据集
    register_dataset()
    
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    # for path in tqdm.tqdm(args.input, disable=not args.output):
    for imgfile in os.listdir(INPUT_IMG_PATH):

        # use PIL, to be consistent with evaluation
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullName, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                imgfile, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(imgfile))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit

