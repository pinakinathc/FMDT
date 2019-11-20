# -*-encoding: utf-8 -*-
import sys
import torch
import torch.nn as nn
from torch.optim import Adadelta

from config import get_args, _C as cfg
from model.backbone import Backbone
from model.semantic import SemanticNet
# from model.geometry import Geometry_Net
from model.model import Model
from dataloader.fashion_dataset import TrainDataset
from lib.nn import user_scattered_collate


def train(args, model, loader_train):
    """
    takes input image as RGB, passes through ConvNet to get backbone feature
    backbone feature is input for semantic segmentation module to get ROI
    the same backbone feature is employed for geometry inference to get a
    global context unlike local context present in semantic segmentation
    TODO: 3 fused feedback iterations from semantic segmentation and global
    geometry inference
    The final output from semantic segmentation is used as ROI
    TODO: ROI region is given to GAN for deformation treatment
    Final ROI is given to text detection

    This module only calls several components to compute the result.
    Each modal returns its own loss value which is combined by this module
    for training the entire fused network.

    Args:

    Returns:
            None.
    """

    model_optim = Adadelta(params=model.parameters(), lr=0.001)

    iterator_train = iter(loader_train)
    for i in range(1):
        batch_data = next(iterator_train)
        print ("keys in batch_data: ", batch_data[0].keys())
        preds = model(batch_data[0], segSize=(256, 256))
        print ("size of preds: ", preds.shape)

    # print ("loss value: ", semantic_loss)


if __name__ == "__main__":
    img_input = torch.randn(2, 3, 256, 256)

    args = get_args(sys.argv[1:])

    dataset_train = TrainDataset(
        cfg.DATASET.image_train,
        cfg.DATASET.json_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)


    device = torch.device(args.cuda)

    backbone_net = Backbone(args.backbone_arch).to(device)
    semantic_net = SemanticNet(args.semantic_arch).to(device)
    critic = nn.NLLLoss(ignore_index=-1)
    model = Model(encoder_net=backbone_net, decoder_net=semantic_net, critic=critic, deep_sup_scale=0.4)

    train(args, model, loader_train)


