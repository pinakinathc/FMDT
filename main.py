# -*-encoding: utf-8 -*-
import sys
import torch
from torch.optim import Adadelta

from config import get_args
from model.backbone import Backbone
from model.semantic import SemanticNet
# from model.geometry import Geometry_Net

args = get_args(sys.argv[1:])

def train(img_input, gt_bounding_box, gt_semantic_label, gt_geometry):
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
                img_input: batch x 256 x 256 x 3
                gt_bounding_box: batch x 4 (bounding box coordinates)
                gt_semantic_label: batch x 256 x 256 x 1 (semantic gt)
                gt_geometry: TODO: Undefined for now
                args: Argument lists to enable training

        Returns:
                None.
        """

        device = torch.device(args.cuda)

        backbone_net = Backbone(args.backbone_arch).to(device)
        backbone_feature = backbone_net(img_input)
        print ("backbone_feature: ", len(backbone_feature))
        print (backbone_feature[0].shape, backbone_feature[1].shape, backbone_feature[2].shape, backbone_feature[3].shape)
        semantic_net = SemanticNet(args.semantic_arch).to(device)
        semantic_out = semantic_net(backbone_feature)

        backbone_optimizer = Adadelta(params=backbone_net.parameters(), lr=args.backbone_lr)
        semantic_optimizer = Adadelta(params=semantic_net.parameters(), lr=args.semantic_lr)

        # semantic_loss.backwards()
        # backbone_optimizer.step()
        # semantic_loss.step()
        print ("semantic feature shape: ", semantic_out[0].shape, semantic_out[1].shape)
        # print ("loss value: ", semantic_loss)


if __name__ == "__main__":
        img_input = torch.randn(2, 3, 256, 256)
        gt_bounding_box = None
        gt_semantic_label = None
        gt_geometry = None

        train(img_input, gt_bounding_box, gt_semantic_label, gt_geometry)

