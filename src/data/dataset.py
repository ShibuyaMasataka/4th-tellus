import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import json
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

import tifffile
from tqdm.auto import tqdm

def random_crop(img, targets, crop_size):  # resize a rectangular image to a padded rectangular
    h, w, _ = img.shape
    
    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    new_img = img[top:bottom, left:right, :]

    i = (targets[:, 0] > left) & (targets[:, 0] < right) & (targets[:, 1] > top) & (targets[:, 1] < bottom)
    new_targets = targets[i]
    new_targets[:, 0] -= left
    new_targets[:, 1] -= top
    return new_img, new_targets

def resize(img, anos, ratio):
	shape = img.shape[:2] 
	new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
	img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
	nL = len(anos)
	if nL > 0:
		anos[:, 0] = ratio * anos[:, 0]
		anos[:, 1] = ratio * anos[:, 1]
	return img, anos

def letterbox(img, height, width):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # padded rectangular
    return img, ratio, dw, dh

def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=0)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:

            # warp points
            n = targets.shape[0]
            xy = np.ones((n, 3))
            xy[:, :2] = targets[:, :]
            xy = (xy @ M.T)

            # reject warped points outside of image
            i = (xy[:, 0] > 0) & (xy[:, 0] < width) & (xy[:, 1] > 0) & (xy[:, 1] < height)

            targets = targets[i]
            targets[:, :] = xy[i, :2]

        return imw, targets, M
    else:
        return imw

def simple_gaussian_radius(det_size):
    height, width = det_size
    rh=0.1155*height
    rw=0.1155*width
    return rh, rw

def simple_gaussian2D(shape, sigmah=1, sigmaw=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x / (2 * sigmaw * sigmaw)+ y * y / (2 * sigmah * sigmah)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def simple_draw_umich_gaussian(heatmap, center, rh, rw, k=1):
    diameterh = 2 * rh + 1
    diameterw = 2 * rw + 1

    gaussian = simple_gaussian2D((diameterh, diameterw), sigmah=diameterh / 6, sigmaw=diameterw / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, rw), min(width - x, rw + 1)
    top, bottom = min(y, rh), min(height - y, rh + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[rh - top:rh + bottom, rw - left:rw + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def get_train_valid_split_data_names(train_img_folder, train_ano_folder, valid_size=1/8):
    # データのパスを学習用と評価用に分割したものを返す
    data_names = []
    for img_name in os.listdir(train_img_folder):
        img_path = train_img_folder + img_name
        ano_path = train_ano_folder + img_name[:-4] + ".json"

        data_names.append({"img_path":img_path, "ano_path":ano_path})
    data_names = np.array(data_names)

    num_valid = int(len(data_names) * valid_size)
    num_train = len(data_names) - num_valid
    num_all = num_train+num_valid
    # replaceがFlaseなら重複なし
    id_all   = np.random.choice(num_all, num_all, replace=False)
    id_valid  = id_all[0:num_valid]
    id_train = id_all[num_valid:num_all]
    valid_data  = data_names[id_valid]
    train_data = data_names[id_train]

    return train_data, valid_data

def get_test_data_names(test_img_folder):
	# データのパスを学習用と評価用に分割したものを返す
    data_names = []
    for img_name in os.listdir(test_img_folder):
        img_path = test_img_folder + img_name

        data_names.append({"img_path":img_path})
    data_names = np.array(data_names)

    return data_names


class Phase1Dataset():  # for training
    def __init__(self, data_names, load_size=(640, 640), augment=False):
        self.num_classes = 1
        self.down_ratio = 4

        self.data = []

        # load images and annotations
        for names in data_names:
            img_path = names["img_path"]
            ano_path = names["ano_path"]

            # 65536で除算して-1～1正規化
            #img = tifffile.imread(img_path)[..., np.newaxis]/65536*2 - 1
            img = np.clip(tifffile.imread(img_path)[..., np.newaxis], 0, 30000)/30000*2 - 1
            ano = []
            with open(ano_path, "r") as f:
                annotations = json.load(f)
                #print(len(annotations['coastline_points']))
                for point in annotations['coastline_points']:
                    ano.append(point)
            ano = np.array(ano)

            self.data.append({"img_path":img_path, "img":img, "ano":ano})

        self.load_width = load_size[0]
        self.load_height = load_size[1]
        self.augment = augment
        self.size = len(self.data)

    def get_data(self, img0, anos0):
        img = img0.copy()
        anos = anos0.copy()
        h, w, _ = img.shape
        #img, anos = resize(img, anos, 1/2)
        #img, anos = random_crop(img[:, :, np.newaxis], anos, (self.load_height, self.load_width))
        img, ratio, padw, padh = letterbox(img, height=self.load_height, width=self.load_width)

        # Load labels
        nL = len(anos)
        if nL > 0:
            anos[:, 0] = ratio * anos[:, 0] + padw
            anos[:, 1] = ratio * anos[:, 1] + padh

        # Augment image and labels
        if self.augment:
            img, anos, M = random_affine(img, anos, degrees=(-180, 180))

            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    anos[:, 0] = self.load_width - anos[:, 0]

        img = torch.from_numpy(img.astype(np.float32)).clone()
        #img = img.unsqueeze(0).permute(0,3,1,2)
        img = img.unsqueeze(0)

        return img, anos, (h, w)

    def __getitem__(self, index):
        #index = index
        if self.augment:
        	index = index // 10
        img_path = self.data[index]["img_path"]
        img = self.data[index]["img"]
        anos = self.data[index]["ano"]

        img, anos, o_size = self.get_data(img, anos)

        output_h = img.shape[1] #// self.down_ratio
        output_w = img.shape[2] #// self.down_ratio
        num_classes = self.num_classes
        num_objs = anos.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        cls_id = 0

        for k in range(num_objs):
            ano = anos[k]
            h = 30
            w = 30

            rh, rw = simple_gaussian_radius((math.ceil(h), math.ceil(w)))
            rh = max(0, int(rh))
            rw = max(0, int(rw))
            #point = np.array([ano[0] // self.down_ratio, ano[1] // self.down_ratio], dtype=np.float32)
            point = np.array([ano[0], ano[1]], dtype=np.float32)
            point_int = point.astype(np.int32)
            simple_draw_umich_gaussian(hm[cls_id], point_int, rh, rw)

        ret = {'img_path':img_path, 'input': img, 'hm': hm}
        return ret

    def __len__(self):
        if self.augment:
        	return self.size * 10
        return self.size

class TestDataset():  # for training
    def __init__(self, data_names, load_size=(640, 640)):
        self.num_classes = 1

        self.data = []

        # load images and annotations
        for names in data_names:
            img_path = names["img_path"]

            # 65536で除算して-1～1正規化
            #img = tifffile.imread(img_path)[..., np.newaxis]/65536*2 - 1
            img = np.clip(tifffile.imread(img_path)[..., np.newaxis], 0, 30000)/30000*2 - 1

            self.data.append({"img_path":img_path, "img":img})

        self.load_width = load_size[0]
        self.load_height = load_size[1]
        self.size = len(self.data)

    def get_data(self, img0):
        img = img0.copy()
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.load_height, width=self.load_width)

        img = torch.from_numpy(img.astype(np.float32)).clone()
        #img = img.unsqueeze(0).permute(0,3,1,2)
        img = img.unsqueeze(0)

        return img, ratio, padw, padh

    def __getitem__(self, index):
        img_path = self.data[index]["img_path"]
        oimg = self.data[index]["img"]

        img, ratio, padw, padh = self.get_data(oimg)

        ret = {'img_path':img_path, 'input': img, 'ratio':ratio, 'padw':padw, 'padh':padh, 'oimg':oimg}
        return ret

    def __len__(self):
        return self.size