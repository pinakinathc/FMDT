# -*- encoding: utf-8 -*-
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/"
_C.cuda = "cuda" # convert to "cuda" during training using GPU

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.SEMANTIC = CN()
_C.DATASET.SEMANTIC.image_train = "./../validation/image/"
_C.DATASET.SEMANTIC.json_train = "./../validation//annos/"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.num_class = 1

_C.DATASET.TEXT = CN()
_C.DATASET.TEXT.trainroot = './../icdar15/trainset/'
_C.DATASET.TEXT.testroot = './../icdar15/testset/'
# multiscale train/test, size of short edge (int or tuple)
# _C.DATASET.imgSizes = (300, 375, 450, 525, 600)
_C.DATASET.imgSizes = (300)
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
_C.MODEL.arch_backbone = "resnet18dialated"
# architecture of net_decoder
_C.MODEL.arch_semantic = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 512
# number of class in model
_C.MODEL.num_class = 2

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 3
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_iters = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.end_iters = 70000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_backbone = 0.02
_C.TRAIN.lr_semantic = 0.02
_C.TRAIN.lr_textdetector = 0.02
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
_C.TRAIN.workers = 15 # Multi-Processing failed for Windows system. Could be increased in other systems

# frequency to display
_C.TRAIN.disp_iter = 20
# frequency to save
_C.TRAIN.checkpoint = 1729
# manual seed
_C.TRAIN.seed = 304

# Episode Semantic
_C.TRAIN.semantic_episode = 73
# Episode Text Detector
_C.TRAIN.text_episode = 73
# resume training from last iteration
_C.TRAIN.resume_iter = 27664 # Enter 0 if starting from scratch
# eval Trained module
_C.TRAIN.eval_iter = 1729

# PSENet Text Detector
_C.TRAIN.TEXT = CN()
_C.TRAIN.TEXT.Lambda = 0.7
_C.TRAIN.TEXT.OHEM_ratio = 3
_C.TRAIN.TEXT.n = 6
_C.TRAIN.TEXT.m = 0.5
_C.TRAIN.TEXT.batch_size = 7
_C.TRAIN.TEXT.workers = 2 # Multi-Processing failed for Windows system. Could be increased in other systems
_C.TRAIN.TEXT.data_shape = 640 

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
