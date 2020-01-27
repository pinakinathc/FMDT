import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, backbone_net,
        semantic_net, textdetector_net, critic, deep_sup_scale=None):

        super(Model, self).__init__()
        self.backbone_net = backbone_net
        self.semantic_net = semantic_net
        self.textdetector_net = textdetector_net
        self.critic_semantic, self.critic_textdetector = critic
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, segSize=None, mode='textdetector'):
        """Argument:
        mode (string): sets the mode of execution of network.
                There are 3 modes available:
                (1): `combined`: employes semantic segmentation + text detector
                (2): `textdetector`: employs text detector Only
                (3): `semantic`: employs semantic segmentation Only
        """
        _, _, imgH, imgW = feed_dict['img_data'].shape

        # for training
        backbone_features = self.backbone_net(feed_dict['img_data'], return_feature_maps=True)

        if mode == "textdetector":
            pred_textdetector = self.textdetector_net(backbone_features, imgH, imgW)
            if self.training: 
                loss_text = self.critic_textdetector(pred_textdetector,
                    feed_dict['text_score'], feed_dict['text_mask'])
                return loss_text[-1], None, loss_text[-1], None, None, None, None
            # raise ValueError ("not implemented") # TODO
            return pred_textdetector

        if mode == "semantic" or mode == "combined":
            if segSize is None:
                loss_text = None
                if self.deep_sup_scale is not None: # use deep supervision technique
                    (pred_semantic, pred_deepsup) = self.semantic_net(backbone_features)
                else:
                    pred_semantic = self.semantic_net(backbone_features)
    
                semantic_loss = self.critic_semantic(pred_semantic, feed_dict['seg_label'])
                total_loss = semantic_loss
                if self.deep_sup_scale is not None:
                    loss_deepsup = self.critic_semantic(pred_deepsup, feed_dict['seg_label'])
                    semantic_loss = semantic_loss + loss_deepsup * self.deep_sup_scale
                    total_loss = semantic_loss
    
                acc = self.pixel_acc(pred_semantic, feed_dict['seg_label'])

                if mode == "combined":
                    raise ValueError("please uncomment this line to execute combined") # TODO
                    fused_data = feed_dict['img_data'] * F.interpolate(
                            pred_semantic[:,-1,:,:].unsqueeze(1).repeat(1,3,1,1),
                            size=(imgH, imgW), mode='bilinear', align_corners=False)
                    
                    new_backbone_features = self.backbone_net(fused_data, return_feature_maps=True)
                    pred_textdetector = self.textdetector_net(new_backbone_features, imgH, imgW)
                    loss_text = self.critic_textdetector(pred_textdetector,
                        feed_dict['text_score'], feed_dict['text_mask'])
                    total_loss = semantic_loss + loss_text
                    
                # return loss, acc
                return total_loss, semantic_loss, None, acc, None, None, None
            # inference
            else:
                new_backbone_features = []
                pred_semantic = None
                pred_semantic = self.semantic_net(backbone_features, segSize=segSize)
                new_backbone_features = []

                for feature in backbone_features:
                    _, channels, ht, wd = feature.shape

                    new_backbone_features.append(feature * F.interpolate(
                        pred_semantic[:,-1,:,:].unsqueeze(1).repeat(1, channels, 1, 1),
                        size=(ht, wd), mode='bilinear', align_corners=False))

                pred_textdetector = self.textdetector_net(backbone_features, imgH, imgW)
                pred_textdetector_new = self.textdetector_net(new_backbone_features, imgH, imgW)

                return pred_semantic, pred_textdetector, pred_textdetector_new

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
