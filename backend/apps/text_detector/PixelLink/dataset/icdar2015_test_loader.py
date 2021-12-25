# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
from pixel_link import cal_gt_for_single_image
import config
#from apps.curvilinear_projection.curvilinearProjection import dewarp
# ic15_root_dir = '/home/gem/phucph/PixelLink.pytorch/dataset/
ic15_root_dir = 'dataset/'
ic15_test_data_dir = ic15_root_dir + 'diff_images/'
# ic15_test_gt_dir = ic15_root_dir + 'test_gt/'

random.seed(123456)

def get_img(img_path, check_dewarp):
    try:
        if check_dewarp == True:
        # img = cv2.imread(img_path)
            img = dewarp(img_path)
            img = img[:, :, [2, 1, 0]]

        else:
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]] 
    except Exception as e:
        print(img_path)
        raise
    return img 

# def scale(img, long_size=2480):
#     h, w = img.shape[0:2]
#     if max(h,w) > long_size:
#         scale = long_size  * 1.0 / max(h, w)
#         h,w=h*scale,w*scale
        
#     sw=int(w/32)*32
#     sh=int(h/32)*32
#     img = cv2.resize(img, dsize=(sw,sh))
#     return img

def scale(image, long_size = 2480):
    ori_h, ori_w = image.shape[0:2]
    scale = 1
    
    if max(ori_h,ori_w) <= long_size:
        scaled_img = image
        
    else :    
        # Resize image to longer size
        scale = long_size*1.0 / max(ori_h,ori_w)
        scaled_img = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        

    padding_w = long_size - scaled_img.shape[1]
    padding_h = long_size - scaled_img.shape[0]
    # color = [255,255,255]
    # color = [0,0,0]
    scaled_padded_image = cv2.copyMakeBorder(scaled_img, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT,value=config.color_padding)
    print("shape:",scaled_padded_image.shape)
    return scaled_padded_image, scaled_img, scale

class IC15TestLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240):
        data_dirs = [ic15_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
            
            self.img_paths.extend(img_paths)

        part_size = int(len(self.img_paths) / part_num)
        l = part_id * part_size
        r = (part_id + 1) * part_size
        self.img_paths = self.img_paths[l:r]
        self.long_size = config.test_long_size # config longsize test

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img= get_img(img_path, False)
        # print("size:",height,weight)
        scaled_padded_image, scaled_img,_scale = scale(img, self.long_size)
        # cv2.imwrite('outputs/' + img_path.split('/')[-1], scaled_img)
        scaled_padded_image = Image.fromarray(scaled_padded_image)
        scaled_padded_image = scaled_padded_image.convert('RGB')
        # scaled_padded_image = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(scaled_padded_image)
        scaled_padded_image = transforms.ToTensor()(scaled_padded_image)
        scaled_padded_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_padded_image)
        
        return img[:, :, [2, 1, 0]], scaled_padded_image, scaled_img, _scale
