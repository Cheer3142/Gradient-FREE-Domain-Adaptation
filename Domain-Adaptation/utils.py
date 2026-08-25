from argparse import Namespace

import logging
import os
from typing import Literal

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

def get_domain_args(
    domain_name: str,
    split: Literal['train', 'val'],
    mode: str = "lora",
    base_model_path: str = "models/model_final.pth",
    num_gpus: int = 1,
    get_cofing_only: bool = False,
):
    logger_names = ["detectron2", "d2", "fvcore"]
    for name in logger_names:
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()

    parts = domain_name.split("-")

    split_list = ["", "", ""]

    for i, part in enumerate(parts):
        split_list[i] = part

    dataset, domain, sub_domain = split_list

    # Supported configurations
    MODE_CHECK = {"lora"}

    DATASET_CHECK = {
        "ACDC",
        "ADE20K",
        "CS",
        "MUSES",
        "MV",
        "BDD",
    }

    # "cs",
    # "acdc",
    # "muses",
    # "bdd",
    # "mv",
    # "a150",
    # "idd",
    # "pc59",
    # "nyu",
    # "coconutL"

    ACDC_DOMAIN_CHECK = {"fog", "night", "snow", "rain"}
    MUSES_DOMAIN_CHECK = {"clear", "rain", "fog", "snow"}
    
    # Configurations assertions
    assert dataset in DATASET_CHECK

    if dataset == "MUSES":
        assert (
            domain in MUSES_DOMAIN_CHECK
        ), f"Domain '{domain}' not supported for MUSES"
    elif dataset == "ACDC":
        assert (
            domain in ACDC_DOMAIN_CHECK
        ), f"Domain '{domain}' is not supported for ACDC"
        assert sub_domain == "", "Volume is not supported in ACDC"

    assert mode in MODE_CHECK, "Mode '{mode}' not supported"

    # configs = {
    #     "CS": {
    #         "normal": f"configs/cityscapes/normal/{mode}-{domain}.yaml"
    #     },
    #     "ACDC": {f"{domain}": f"configs/acdc/{domain}/{mode}-{domain}-acdc.yaml"},
    #     "muses": {
    #         f"{domain}": f"configs/muses/{domain}/muses-{domain}-{sub_domain}.yaml"
    #     },
    #     "bdd": "configs/bdd/bdd.yaml",
    #     "mv": "configs/mv/mv.yaml",
    #     "nyu": "configs/nyu/nyu.yaml",
    #     "a150": "configs/a150/a150.yaml",
    #     "idd": "configs/idd/idd.yaml",
    #     'pc59': 'configs/pc59/pc59.yaml',
    #     'nyu': 'configs/nyu/nyu.yaml',
    #     'coconutL': 'configs/coconutL/coconutL.yaml'
    # }


    dataset_main_path = "/home/chenyinjia/data/dataset/"
    datasets = {
        "ADE20K": {
            "train": f"{dataset_main_path}ADE20ID/images/training/",
            "val": f"{dataset_main_path}ADE20ID/images/validation/",
        },
        "CS": {
            "train": f"{dataset_main_path}cityscapes/leftImg8bit/train/",
            "val": f"{dataset_main_path}cityscapes/leftImg8bit/val/",
        },
        "ACDC": {
            "train": f"{dataset_main_path}acdc/rgb_anon/{domain}/train/",
            "val": f"{dataset_main_path}acdc/rgb_anon/{domain}/val/",
        },
        "MUSES": {
            "train": f"{dataset_main_path}muses/frame_camera/train/{domain}/",
            "val": f"{dataset_main_path}muses/frame_camera/val/{domain}/",
        },
        "BDD": {
            "train": f"{dataset_main_path}bdd100k/images/10k/train/",
            "val": f"{dataset_main_path}bdd100k/images/10k/val/",
        },
        "MV": {
            "train": f"{dataset_main_path}mapillary_vistas/train/images/",
            "val": f"{dataset_main_path}mapillary_vistas/val/images/",
        },
        # No Dataset Yet

        # "idd": {
        #     "train": f"{dataset_main_path}IDD_Segmentation/leftImg8bit/train/",
        #     "val": f"{dataset_main_path}IDD_Segmentation/leftImg8bit/val/",
        # },
        # "pc59": {
        #     "train": f"{dataset_main_path}pascal_ctx_d2/images/training",
        #     "val": f"{dataset_main_path}pascal_ctx_d2/images/validation",
        # },
        # "nyu": {
        #     "train": f"{dataset_main_path}nyudv2_splitted/train/rgb",
        #     "val": f"{dataset_main_path}nyudv2_splitted/test/rgb",
        # },
        # "coconutL": {
        #     "train": f"{dataset_main_path}coconut-l/train2017/",
        #     "val": f"{dataset_main_path}coconut-l/val2017",
        # },
    }

    # Output path configuration
    output_path = (
        f"output/{dataset}/{mode}-{dataset}"
        + (f"-{domain}" if  domain != "" else "")
        + (f"-{sub_domain}" if sub_domain != "" else "")
        + "/eval/"
    )

    # print(output_path)

    # # Constructing the return values
    # if domain == "" and sub_domain == "":
    #     config_file = configs[dataset]
    # else:
    #     config_file = configs[dataset][domain]

    train_dataset_path = (
        datasets[dataset]["train"]
        if dataset != "cs"
        else datasets[dataset][domain]["train"]
    )

    val_dataset_path = (
        datasets[dataset]["val"]
        if dataset != "cs"
        else datasets[dataset][domain]["val"]
    )

    args = Namespace(
        # config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            base_model_path,
        ],
        resume=True,
    )

    if get_cofing_only:
        return args


def custom_domain_args(
    config_file,
    output_path,
    num_gpus=1,
    model_path="models/model_final.pth",
    dataset_path: str = None,
    seed=None,
):

    args = Namespace(
        config_file=config_file,
        eval_only=True,
        num_gpus=num_gpus,
        dataset_path=dataset_path,
        opts=[
            "OUTPUT_DIR",
            output_path,
            "TEST.SLIDING_WINDOW",
            "True",
            "MODEL.SEM_SEG_HEAD.POOLING_SIZES",
            "[1,1]",
            "MODEL.WEIGHTS",
            model_path,
        ],
        resume=True,
        model_path=model_path,
    )

    if seed != None:
        args.opts.extend(["SEED", seed])

    return args


def benchmark_catseg(model, args):

    import detectron2.utils.comm as comm
    from detectron2.evaluation import verify_results

    from catseg.train_net import Trainer, set_random_seed, setup

    cfg = setup(args)
    set_random_seed(cfg.SEED)
    res = Trainer.test(cfg, model)
    if cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(cfg, model))
    if comm.is_main_process():
        verify_results(cfg, res)
    return res

def load_catseg_model(args, model_path: str = None):
    from catseg.train_net import Trainer, setup
    from detectron2.checkpoint import DetectionCheckpointer

    print("Loading base model ...")
    
    try:
        cfg = setup(args)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS if model_path is None else model_path, resume=args.resume
        )
        print("Base model loaded.\n")
        return model
    except AttributeError as e:
        print(f"Error: Invalid model configuration: {e}")
        raise
    except FileNotFoundError:
        print(f"Error: Model weights not found at the specified path.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading the base model: {e}")
        raise