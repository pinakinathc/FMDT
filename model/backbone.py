# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision

from . import resnet


class Backbone(nn.Module):
	def __init__(self, cfg):
		super(Backbone, self).__init__()
		if cfg.MODEL.arch_backbone == "resnet18dialated":
			# self.network = ResNet18()
			orig_resnet = resnet.__dict__['resnet18'](pretrained=True)
			self.network = ResnetDilated(orig_resnet, dilate_scale=8)
		else:
			raise ValueError("please specify backbone architecture. Recieve args as: "+cfg.MODEL.arch_backbone)

	def forward(self,input_, return_feature_maps=True):
		out = self.network(input_, return_feature_maps=return_feature_maps)
		return out


class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18, self).__init__()

		resnet18_original = torchvision.models.resnet18(pretrained=True)
		layers = list(resnet18_original.children())[:-2] # removing the last fc layer and avg pool
		self.resnet18 = nn.Sequential(*layers)

	def forward(self, x):
		return self.resnet18(x)


class ResnetDilated(nn.Module):
	def __init__(self, orig_resnet, dilate_scale=8):
		super(ResnetDilated, self).__init__()
		from functools import partial

		if dilate_scale == 8:
			orig_resnet.layer3.apply(
				partial(self._nostride_dilate, dilate=2))
			orig_resnet.layer4.apply(
				partial(self._nostride_dilate, dilate=4))
		elif dilate_scale == 16:
			orig_resnet.layer4.apply(
				partial(self._nostride_dilate, dilate=2))

		# take pretrained resnet, except AvgPool and FC
		self.conv1 = orig_resnet.conv1
		self.bn1 = orig_resnet.bn1
		self.relu1 = orig_resnet.relu1
		self.conv2 = orig_resnet.conv2
		self.bn2 = orig_resnet.bn2
		self.relu2 = orig_resnet.relu2
		self.conv3 = orig_resnet.conv3
		self.bn3 = orig_resnet.bn3
		self.relu3 = orig_resnet.relu3
		self.maxpool = orig_resnet.maxpool
		self.layer1 = orig_resnet.layer1
		self.layer2 = orig_resnet.layer2
		self.layer3 = orig_resnet.layer3
		self.layer4 = orig_resnet.layer4

	def _nostride_dilate(self, m, dilate):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			# the convolution with stride
			if m.stride == (2, 2):
				m.stride = (1, 1)
				if m.kernel_size == (3, 3):
					m.dilation = (dilate//2, dilate//2)
					m.padding = (dilate//2, dilate//2)
			# other convoluions
			else:
				if m.kernel_size == (3, 3):
					m.dilation = (dilate, dilate)
					m.padding = (dilate, dilate)

	def forward(self, x, return_feature_maps=False):
		conv_out = []

		x = self.relu1(self.bn1(self.conv1(x)))
		x = self.relu2(self.bn2(self.conv2(x)))
		x = self.relu3(self.bn3(self.conv3(x)))
		x = self.maxpool(x)

		x = self.layer1(x); conv_out.append(x);
		x = self.layer2(x); conv_out.append(x);
		x = self.layer3(x); conv_out.append(x);
		x = self.layer4(x); conv_out.append(x);

		if return_feature_maps:
			return conv_out
		return [x]
