# -*- encoding: utf-8 -*-
import numpy as np
import os
import cv2
import torch
from torchvision import transforms
from model.backbone import Backbone
from model.semantic import SemanticNet
from model.psenet import PSENet
from model.model import Model
from main import load_model_weights
from pse import decode as pse_decode
from config import _C as cfg
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates results from the fused network")
    parser.add_argument('--testdir', type=str, help='enter directory containing test images')
    parser.add_argument('--long_size', type=int, default=750, help='size of the longest side of image')
    args = parser.parse_args()

    device = torch.device(cfg.cuda)
    backbone_net = Backbone(cfg).to(device)
    semantic_net = SemanticNet(cfg, use_softmax=True).to(device)
    pse_net = PSENet()
    model = Model(backbone_net=backbone_net, semantic_net=semantic_net,
        textdetector_net=pse_net,
        critic=[None, None],
        deep_sup_scale=cfg.TRAIN.deep_sup_scale).to(device)
    model, _ = load_model_weights(model, cfg, cfg.TRAIN.resume_iter)
    
    for root, _, files in os.walk(args.testdir):
        for file in files:
            img_path = os.path.join(root, file)
            img_original = cv2.imread(img_path)
            if img_original is None:
                continue
            h, w = img_original.shape[:2]
            segSize = (h, w)
            scale = args.long_size / max(h, w)
            # scale = 1
            img = img_original.copy()
            print ("img_original shape: ", img_original.shape)
            img = cv2.resize(img, None, fx=scale, fy=scale).astype(np.float32)
            print ("img_original shape: {}, img shape: {}".format(img_original.shape, img.shape))
            tensor_img = transforms.ToTensor()(img)
            tensor_img = tensor_img.unsqueeze(0)
            tensor_img = tensor_img.repeat(2, 1, 1, 1)
            tensor_img = tensor_img.to(device)
            print ("segSize: ", segSize)
            feed_dict = {'img_data': tensor_img, 'seg_label': None, 'text_score': None, 'text_mask': None}
            with torch.no_grad():
                pred_semantic, pred_textdetector, pred_textdetector_atten = model(feed_dict, segSize=segSize, mode='combined')

                pred_textdetector, boxes_list = pse_decode(pred_textdetector[0], 1)
                pred_textdetector_atten, boxes_list_atten = pse_decode(pred_textdetector_atten[0], 1)

                scale = (pred_textdetector.shape[1]*1.0/w, pred_textdetector.shape[0]*1.0/h)
                if len(boxes_list):
                    boxes_list = boxes_list / scale
                if len(boxes_list_atten):
                    boxes_list_atten = boxes_list_atten / scale

            img_textdetector = img_original.copy()
            img_textdetector_atten = img_original.copy()

            cv2.polylines(img_textdetector, boxes_list.astype(np.int32), True, (0, 255, 0), thickness=3)
            cv2.polylines(img_textdetector_atten, boxes_list_atten.astype(np.int32), True, (0, 255, 0), thickness=3)

            pred_semantic = (pred_semantic[0].permute(1,2,0)).cpu().numpy()
            plt.imshow(pred_semantic[:,:,1], cmap='hot')
            plt.show()
            pred_semantic[pred_semantic>0.5] = 1
            pred_semantic[pred_semantic!=1] = 0
            masked_img = np.repeat(pred_semantic[:,:,1][:,:,np.newaxis], 3, -1).astype(np.uint8)
            masked_img *= img_original

            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(img_original[:,:,::-1])
            ax[0,0].set_title("input image")
            ax[0,1].imshow(masked_img[:,:,::-1])
            ax[0,1].set_title("cropped region with >50% confidence")
            ax[1,0].imshow(img_textdetector[:,:,::-1])
            ax[1,0].set_title("text on original input image")
            ax[1,1].imshow(img_textdetector_atten[:,:,::-1])
            ax[1,1].set_title("text after attention")
            # ax[2].imshow(pred_semantic[:, :, 1])
            # plt.show()
            new_root = os.path.join("..", "output", "pinaki", root)
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            plt.savefig(os.path.join(new_root, file[:-3]+"png"))
