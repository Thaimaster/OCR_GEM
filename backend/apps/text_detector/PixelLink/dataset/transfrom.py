import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from shapely.geometry import box, Polygon
import math
from .image_process import *
import config
from scipy.misc import imread, imresize


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, tags=None):
        for t in self.transforms:
            img, boxes, tags = t(img, boxes, tags)
        return img, boxes, tags


class Resize_Padding(object):
    def __init__(self, size=(1280, 1280)):
        self.width = size[1]
        self.heigth = size[0]
        self.long_size = config.train_long_size

    def __call__(self, image, boxes=None, tags=None):
        ori_h, ori_w, c = image.shape
        # delta_w = 0
        # delta_h = 0
        if max(ori_h,ori_w) <= self.long_size:
          
          delta_w = 1312 - ori_w if 1280 - ori_w > 0 else 0  
          delta_h = 1312 - ori_h if 1280 - ori_h > 0 else 0    
        # if max(ori_h,ori_w) <= self.long_size:
        #     delta_w = self.width - ori_w
        #     delta_h = self.heigth - ori_h          
           
        if max(ori_h,ori_w) > self.long_size:
            
            # Resize image to longer size
            scale = self.long_size*1.0 / max(ori_h,ori_w)
            image=cv2.resize(image, dsize=None, fx=scale, fy=scale)       
            if boxes is not None: 
                boxes[:, :, 0] *= scale
                boxes[:, :, 1] *= scale
                boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, self.width)
                boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, self.heigth)
            
            # Padding to longer size
            delta_w = self.width - image.shape[1]
            delta_h = self.heigth - image.shape[0]

        
        new_image = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT,value=config.color_padding)
        # cv2.copyMakeBorder(image, 100, 100, 50, 50, cv2.BORDER_REFLECT)
        # bbox= np.array(boxes, dtype=np.int32)    
        return new_image, boxes, tags
            

class RandomResize(object):
    def __init__(self, longer_sides=config.random_resize):
        self.longer_sides = longer_sides

    def __call__(self, image, boxes=None, tags=None):
        return random_resize(image, boxes, tags, self.longer_sides)


class RandomRotate(object):
    def __init__(self, rotate_angles=np.arange(-5, 5, 1)):
        self.rotate_angles = rotate_angles

    def __call__(self, image, boxes=None, tags=None):
        return random_rotate(image, boxes, tags, self.rotate_angles)


class RandomRatioScale(object):
    def __init__(self, random_ratios=np.arange(0.8, 1.3, 0.1)):
        self.random_ratios = random_ratios

    def __call__(self, image, boxes, tags):
        return random_ratio_scale(image, boxes, tags, self.random_ratios)


class RandomCrop(object):
    def __init__(self, crop_size=config.crop_size, max_tries=config.max_tries):
        self.crop_size = crop_size
        self.max_tries = max_tries

    def __call__(self, image, boxes, tags):
        return random_crop(image, boxes, tags, self.crop_size, self.max_tries)


class RandomRotate90(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, image, boxes, tags):
        if random.random() > self.ratio:
            return image, boxes, tags
        h, w, _ = image.shape
        image = np.rot90(image)
        new_boxes = np.zeros_like(boxes)
        for i, box in enumerate(boxes):
            new_boxes[i] = abs(box - [w, 0])
        new_boxes = new_boxes[:, (1, 2, 3, 0), :][:, :, (1, 0)]
        return image, new_boxes, tags


class build_transfrom(object):
    def __init__(self):
        self.augment = Compose([
            RandomResize(),
            # RandomRatioScale(),
            
            RandomRotate90(),
            RandomRotate(),
            RandomCrop()
        ])

    def __call__(self, img, boxes, tags):
        return self.augment(img, boxes, tags)


class SythAugmentation(object):
    def __init__(self):
        self.augment = Compose([
            Resize_Padding(config.train_image_size)
        ])

    def __call__(self, img, boxes, tags):
        return self.augment(img, boxes, tags)
