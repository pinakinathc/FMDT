# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

inplace = True

class PSENet(nn.Module):
    """Perform text detection from backbone features"""
    def __init__(self, result_num=6, scale:int=1, output_filters=[64, 128, 256, 512]):
        """output_filters are predefined according to resnet18 architecture"""

        super(PSENet, self).__init__()
        self.scale = scale
        conv_out = 256

        # Top Layer
        self.toplayer = nn.Sequential(nn.Conv2d(
            output_filters[3], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(
            output_filters[2], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        self.latlayer2 = nn.Sequential(nn.Conv2d(
            output_filters[1], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        self.latlayer3 = nn.Sequential(nn.Conv2d(
            output_filters[0], conv_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        # Smooth layers
        self.smooth1 = nn.Sequential(nn.Conv2d(
            conv_out, conv_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        self.smooth2 = nn.Sequential(nn.Conv2d(
            conv_out, conv_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        self.smooth3 = nn.Sequential(nn.Conv2d(
            conv_out, conv_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_out), nn.ReLU(inplace=inplace))

        self.conv = nn.Sequential(
            nn.Conv2d(conv_out*4, conv_out,
                 kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU(inplace=inplace))

        self.out_conv = nn.Conv2d(conv_out, result_num, kernel_size=1, stride=1)

    def forward(self, input_:torch.Tensor, imgH, imgW):
        c2, c3, c4, c5 = input_
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)

        if self.train:
            x = F.interpolate(x, size=(imgH, imgW),
                mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(imgH // self.scale, 
                imgW // self.scale), mode='bilinear', align_corners=True)

        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], 
            mode='bilinear', align_corners=False) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([p2, p3, p4, p5], dim=1)


if __name__ == "__main__":
    import time
    device = torch.device('cpu')
    net = PSENet().to(device)
    net.eval()
    x = [torch.randn(2, 64, 92, 92), torch.randn(2, 128, 46, 46),
        torch.randn(2, 256, 46, 46), torch.randn(2, 512, 46, 46)]
    start_time = time.time()
    y = net(x, 256, 256)
    print ("time taken: ", time.time()-start_time)
    print ("output_feature map shape: ", y.shape)
