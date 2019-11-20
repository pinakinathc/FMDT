# -*- encoding: utf-8 -*-
import argparse

# parser = argparse.ArgumentParser(description="Encapsulates essential default config values of FMDT")
# parser.add_argument("--train_data_path", type=str, help="path to training directory")
# parser.add_argument("--validation_data_path", type=str, help="path to validation data path")
# parser.add_argument("--evaluate_only", action="store_true", help="flag specifying only evaluation")
# parser.add_argument("--logs_path", type=str, help="directory path of logs for training")
# parser.add_argument("--trained_model_path", type=str, help="directory path for saving trained weights")
# parser.add_argument("--backbone_arch", type=str, default="resnet18", help="convolutional architecture for backbone feature extraction")
# parser.add_argument("--semantic_arch", type=str, default="upernet", help="architecture for semantic segmentation")
# parser.add_argument("--cuda", type=str, default="cpu", help="use cuda or cpu")
# parser.add_argument("--backbone_lr", type=float, default=0.001, help="learning rate for backbone CNN")
# parser.add_argument("--semantic_lr", type=float, default=0.001, help="learning rate for semantic network")

# def get_args(sys_args):
# 	return parser.parse_args(sys_args)


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"
_C.cuda = "cpu" # convert to "gpu" during training

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.image_train = "./../DeepFashionData/train/image/"
_C.DATASET.json_train = "./../DeepFashionData/train/annos/"
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

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet18dialated"
# architecture of net_decoder
_C.MODEL.arch_semantic = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 512

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_iters = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.end_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# frequency to save
_C.TRAIN.checkpoint = 1729
# manual seed
_C.TRAIN.seed = 304

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# the checkpoint to evaluate on
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = "epoch_20.pth"
# folder to output visualization results
_C.TEST.result = "./"