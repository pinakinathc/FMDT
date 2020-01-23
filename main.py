# -*-encoding: utf-8 -*-
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adadelta
from torchvision import transforms

from config import _C as cfg
from model.backbone import Backbone
from model.semantic import SemanticNet
from model.psenet import PSENet
from model.pseloss import PSELoss
# from model.geometry import Geometry_Net
from model.model import Model
from eval_model import eval_model
from dataloader.fashion_dataset import TrainDataset as SemanticDataset
from dataloader.textdetector_dataset import MyDataset as TextDataset
from utils import AverageMeter, setup_logger
from lib.nn import user_scattered_collate


def checkpoint(model, history, cfg, iters):
    print('Saving checkpoints...')

    dict_backbone = model.backbone_net.state_dict()
    dict_semantic = model.semantic_net.state_dict()
    dict_text = model.textdetector_net.state_dict()

    torch.save(
        history,
        '{}/history_iter_{}.pth'.format(cfg.DIR, iters))
    torch.save(
        dict_backbone,
        '{}/backbone_iter_{}.pth'.format(cfg.DIR, iters))
    torch.save(
        dict_semantic,
        '{}/semantic_iter_{}.pth'.format(cfg.DIR, iters))
    torch.save(
        dict_text,
        '{}/textdetector_iter_{}.pth'.format(cfg.DIR, iters))


def load_model_weights(model, cfg, iters):
    print ('loading checkpoint...')
    history = torch.load('{}/history_iter_{}.pth'.format(cfg.DIR, iters))
    model.backbone_net.load_state_dict(torch.load(
        '{}/backbone_iter_{}.pth'.format(cfg.DIR, iters)))
    model.semantic_net.load_state_dict(torch.load(
        '{}/semantic_iter_{}.pth'.format(cfg.DIR, iters)))
    model.textdetector_net.load_state_dict(torch.load(
        '{}/textdetector_iter_{}.pth'.format(cfg.DIR, iters)))
    return model, history


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
    (backbone_net, semantic_net, pse_net) = nets
    optimizer_backbone = torch.optim.SGD(
        group_weight(backbone_net),
        lr=cfg.TRAIN.lr_backbone,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_semantic = torch.optim.SGD(
        group_weight(semantic_net),
        lr=cfg.TRAIN.lr_semantic,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_textdetector = torch.optim.SGD(
        group_weight(pse_net),
        lr=cfg.TRAIN.lr_textdetector,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_backbone, optimizer_semantic, optimizer_textdetector)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.end_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_backbone = cfg.TRAIN.lr_backbone * scale_running_lr
    cfg.TRAIN.running_lr_semantic = cfg.TRAIN.lr_semantic * scale_running_lr
    cfg.TRAIN.running_lr_textdetector = cfg.TRAIN.lr_textdetector * scale_running_lr

    (optimizer_backbone, optimizer_semantic, optimizer_textdetector) = optimizers
    for param_group in optimizer_backbone.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_backbone
    for param_group in optimizer_semantic.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_semantic
    for param_group in optimizer_textdetector.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_textdetector


def train(model, loader_semantic, loader_text, history, device):
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

    cfg.TRAIN.running_lr_backbone = cfg.TRAIN.lr_backbone
    cfg.TRAIN.running_lr_semantic = cfg.TRAIN.lr_semantic
    cfg.TRAIN.running_lr_textdetector = cfg.TRAIN.lr_textdetector

    history = {'train': {'iters': [], 'loss': [], 'semantic_loss': [], 'text_loss': [],
        'acc': [], 'text_loss': [], 'precision': [], 'recall': [], 'hmean': []}}

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_total_semantic_loss = AverageMeter()
    ave_total_text_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_precision = AverageMeter()
    ave_recall = AverageMeter()
    ave_hmean = AverageMeter()

    if cfg.TRAIN.resume_iter:
        model, history = load_model_weights(model, cfg, cfg.TRAIN.resume_iter)
        cfg.TRAIN.start_iters = cfg.TRAIN.resume_iter
        adjust_learning_rate(optimizers, cfg.TRAIN.start_iters, cfg)

    iterator_semantic = iter(loader_semantic)
    iterator_text = iter(loader_text)

    tic = time.time()

    semantic_iter_count = 0; text_iter_count = 0; mode=None # TODO: remove this fixed value
    for cur_iter in range(cfg.TRAIN.start_iters, cfg.TRAIN.end_iters):

        if cur_iter % cfg.TRAIN.eval_iter == 0:
            eval_model(model, "./output/", cfg.DATASET.TEXT.testroot, device)
            model.train()

        if semantic_iter_count < cfg.TRAIN.semantic_episode:
            batch_semantic_data = next(iterator_semantic)
            img_data = batch_semantic_data[0]["img_data"]
            seg_label = batch_semantic_data[0]["seg_label"]
            text_score = torch.tensor(float('NaN'))
            text_mask = torch.tensor(float('NaN'))
            semantic_iter_count += 1
            mode = 'semantic'
        elif text_iter_count < cfg.TRAIN.text_episode:
            try:
                batch_text_data = next(iterator_text)
            except:
                iterator_text = iter(loader_text)
                batch_text_data = next(iterator_text)
            img_data = batch_text_data[0]
            seg_label = torch.tensor(float('NaN'))
            text_score = batch_text_data[1]
            text_mask = batch_text_data[2]
            text_iter_count += 1
            mode = 'textdetector'
        else:
            semantic_iter_count = 0; text_iter_count = 0
            
        data_time.update(time.time() - tic)
        model.zero_grad()
        
        adjust_learning_rate(optimizers, cur_iter, cfg)

        batch_data = {"img_data": img_data.to(device),
            "seg_label": seg_label.to(device),
            "text_score": text_score.to(device),
            "text_mask": text_mask.to(device)}

        loss, semantic_loss, text_loss, acc, precision, recall, hmean = model(batch_data, mode=mode)
        loss = loss.mean()

        # acc = acc.mean()
        semantic_loss = semantic_loss.mean() if semantic_loss is not None else None
        text_loss = text_loss.mean() if text_loss is not None else None
        acc = acc.mean() if acc is not None else None
        precision = precision.mean() if precision is not None else None
        recall = recall.mean() if recall is not None else None
        hmean = hmean.mean() if hmean is not None else None

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())

        if semantic_loss is not None:
            ave_total_semantic_loss.update(semantic_loss.data.item())
        if text_loss is not None:
            ave_total_text_loss.update(text_loss.data.item())
        if acc is not None:
            ave_acc.update(acc.data.item()*100)
        if precision is not None:
            ave_precision.update(precision.data.item())
        if recall is not None:
           ave_recall.update(recall.data.item())
        if hmean is not None:
           ave_hmean.update(hmean.data.item())

        # calculate accuracy, and display
        if cur_iter % cfg.TRAIN.disp_iter == 0:
            print('Iters: [{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_backbone: {:.6f}, lr_semantic: {:.6f}, lr_text: {:.6f}, '
                  'Loss: {:.6f}, Semantic_Loss: {:.3f}, Text_Loss: {:.3f}, '
                  'Semantic_Accuracy: {:4.2f}'
                  .format(cur_iter, cfg.TRAIN.end_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_backbone, cfg.TRAIN.running_lr_semantic,
                          cfg.TRAIN.running_lr_textdetector,
                          ave_total_loss.average(), ave_total_semantic_loss.average(),
                          ave_total_text_loss.average(), ave_acc.average()))

            history['train']['iters'].append(cur_iter)
            history['train']['loss'].append(loss.data.item())
            if acc is not None:
                history['train']['acc'].append(acc.data.item())

        if cur_iter % cfg.TRAIN.checkpoint == 0:
            checkpoint(model, history, cfg, cur_iter)
        

if __name__ == "__main__":
    device = torch.device(cfg.cuda)

    dataset_semantic = SemanticDataset(
        cfg.DATASET.SEMANTIC.image_train,
        cfg.DATASET.SEMANTIC.json_train,
        cfg.DATASET,
        device,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_semantic = torch.utils.data.DataLoader(
        dataset_semantic,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.TRAIN.workers,
        collate_fn=user_scattered_collate,
        drop_last=True,
        pin_memory=True)

    dataset_textdetector = TextDataset(cfg.DATASET.TEXT.trainroot,
        data_shape=cfg.TRAIN.TEXT.data_shape, n=cfg.TRAIN.TEXT.n,
        m=cfg.TRAIN.TEXT.m, transform=transforms.ToTensor())

    loader_text = torch.utils.data.DataLoader(
        dataset_textdetector,
        batch_size=cfg.TRAIN.TEXT.batch_size,
        shuffle=True,
        num_workers=cfg.TRAIN.TEXT.workers)

    backbone_net = Backbone(cfg).to(device)
    semantic_net = SemanticNet(cfg).to(device)
    pse_net = PSENet()
    critic_semantic = nn.NLLLoss(ignore_index=-1)
    critic_textdetector = PSELoss(Lambda=cfg.TRAIN.TEXT.Lambda,
        ratio=cfg.TRAIN.TEXT.OHEM_ratio, reduction='mean')

    model = Model(backbone_net=backbone_net, semantic_net=semantic_net,
        textdetector_net=pse_net,
        critic=[critic_semantic, critic_textdetector],
        deep_sup_scale=cfg.TRAIN.deep_sup_scale).to(device)

    optimizers = create_optimizers((backbone_net, semantic_net, pse_net), cfg)

    train(model, loader_semantic, loader_text, optimizers, device)


