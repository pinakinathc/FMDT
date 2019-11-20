import os
import json
import torch
from torchvision import transforms
import numpy as np 
from PIL import Image, ImageDraw


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST 
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def get_segm(original_img, segm_file):
    img = np.zeros_like(np.asarray(original_img)) # draw black canvas
    with open(segm_file) as json_data:
        d = json.load(json_data)
        json_data.close()

    keys = d.keys()
    for key in keys:
        if 'item' in key:
            polygons = d[key]['segmentation']
            for polygon in polygons:
                x = map(int, polygon[::2])
                y = map(int, polygon[1::2])
                # print (list(zip(x,y)))
                img = Image.fromarray(img)
                draw = ImageDraw.Draw(img)
                draw.polygon(list(zip(x,y)), fill="white")
                img = np.asarray(img)
    return Image.fromarray(img).convert('L')


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, segm_odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, segm_odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def get_samples(self, odgt, segm_odgt):
        list_sample = []
        for root, dirs, files in os.walk(odgt):
            for file in files:
                list_sample.append([os.path.join(root, file), 
                    os.path.join(segm_odgt, file[:-4]+".json")])
        return list_sample

    def parse_input_list(self, odgt, segm_odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = self.get_samples(odgt, segm_odgt)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = np.array(segm)
        segm[segm!=255] = 1
        segm[segm==255] = 2
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, image_dataset, json_dataset, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(image_dataset, json_dataset, opt, **kwargs)
        self.image_dataset = image_dataset
        self.json_dataset = json_dataset
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        #classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = []

        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            self.batch_record_list.append(this_sample)

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list) == self.batch_per_gpu:
                batch_records = self.batch_record_list
                self.batch_record_list = []
                break
        return batch_records

    def __getitem__(self, index):
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffle = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all image' to a chisen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        images = []
        segm_images = []
        for i in range(self.batch_per_gpu):
            img = Image.open(batch_records[i][0]).convert('RGB')
            segm = get_segm(img, batch_records[i][1])
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])            
            img_width, img_height = img.size
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale
            images.append(img)
            segm_images.append(segm)

        # We shall pad both input image and segmentation maps to size h' and w'
        # so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            img = images[i]
            segm = segm_images[i]
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(img)
            # ax[1].imshow(segm)
            # plt.show()
            # print ("hey!!!!!!!!!!!!!!")

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass
