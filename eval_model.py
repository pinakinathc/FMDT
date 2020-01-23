from tqdm import tqdm
import cv2
import os
import shutil
import numpy as np
import torch
from torchvision import transforms
from pse import decode as pse_decode
from cal_recall import cal_recall_precison_f1


def eval_model(model, save_path, test_path, device, config_scale=1):
    print ("evaluating model...please check outputs directory after this...")
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        img_original = img.copy()
        h, w = img.shape[:2]
        #if max(h, w) > long_size:
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        feed_dict = {'img_data': tensor, 'seg_label': None, 'text_score': None, 'text_mask': None}
        with torch.no_grad():
            preds = model(feed_dict, segSize=None, mode='textdetector')
            preds, boxes_list = pse_decode(preds[0], config_scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale

        cv2.polylines(img_original, boxes_list.astype(np.int32), True, (0, 255, 0))
        cv2.imwrite(save_name[:-3]+'png', img_original)
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict
    print ('result: ', result_dict)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']

