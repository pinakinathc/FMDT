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
from utils import AverageMeter, setup_logger


def checkpoint(model, history, cfg, iters):
    print('Saving checkpoints...')

    dict_backbone = model.backbone_net.state_dict()
    dict_semantic = model.semantic_net.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, iters))
    torch.save(
        dict_backbone,
        '{}/backbone_epoch_{}.pth'.format(cfg.DIR, iters))
    torch.save(
        dict_semantic,
        '{}/semantic_epoch_{}.pth'.format(cfg.DIR, iters))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (backbone_net, semantic_net) = nets
    optimizer_backbone = torch.optim.SGD(
        group_weight(backbone_net),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_semantic = torch.optim.SGD(
        group_weight(semantic_net),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_backbone, optimizer_semantic)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def train(model, loader_train, history):
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

    history = {'train': {'iters': [], 'loss': [], 'acc': []}}

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    iterator_train = iter(loader_train)

    tic = time.time()
    for cur_iter in range(cfg.TRAIN.start_iters, cfg.TRAIN.end_iters):
        batch_data = next(iterator_train)
        data_time.update(time.time() - tic)
        model.zero_grad()
        
        adjust_learning_rate(optimizers, cur_iter, cfg)

        loss, acc = model(batch_data[0])
        loss = loss.mean()
        acc = acc.mean()

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if cur_iter % cfg.TRAIN.disp_iter == 0:
            print('Iters: [{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(cur_iter, cfg.TRAIN.end_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['iters'].append(cur_iter)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

        if cur_iter % cfg.TRAIN.checkpoint == 0:
            checkpoint((model.backbone_net, model.semantic_net), history, cfg)
        

if __name__ == "__main__":
    img_input = torch.randn(2, 3, 256, 256)

    dataset_train = TrainDataset(
        cfg.DATASET.image_train,
        cfg.DATASET.json_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=False,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)


    device = torch.device(cfg.cuda)

    backbone_net = Backbone(cfg.arch_encoder).to(device)
    semantic_net = SemanticNet(cfg.arch_semantic).to(device)
    critic = nn.NLLLoss(ignore_index=-1)
    model = Model(encoder_net=backbone_net, decoder_net=semantic_net, critic=critic, deep_sup_scale=cfg.TRAIN.deep_sup_scale)

    optimizers = create_optimizers((backbone_net, semantic_net), cfg)

    train(model, loader_train, optimizers)


