# -*- encoding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description="Encapsulates essential default config values of FMDT")
parser.add_argument("--train_data_path", type=str, help="path to training directory")
parser.add_argument("--validation_data_path", type=str, help="path to validation data path")
parser.add_argument("--evaluate_only", action="store_true", help="flag specifying only evaluation")
parser.add_argument("--logs_path", type=str, help="directory path of logs for training")
parser.add_argument("--trained_model_path", type=str, help="directory path for saving trained weights")
parser.add_argument("--backbone_arch", type=str, default="resnet18", help="convolutional architecture for backbone feature extraction")
parser.add_argument("--semantic_arch", type=str, default="upernet", help="architecture for semantic segmentation")
parser.add_argument("--cuda", type=str, default="cpu", help="use cuda or cpu")
parser.add_argument("--backbone_lr", type=float, default=0.001, help="learning rate for backbone CNN")
parser.add_argument("--semantic_lr", type=float, default=0.001, help="learning rate for semantic network")

def get_args(sys_args):
	return parser.parse_args(sys_args)


from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.image_train = "./../train/image/"
_C.DATASET.json_train = "./../train/annos/"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.num_class = 1
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
_C.TRAIN.workers = 2
