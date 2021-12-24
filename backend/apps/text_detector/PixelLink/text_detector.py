import torch
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import cv2
import inspect
import sys
import torchvision.transforms as transforms
import collections

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
sys.path.insert(0, currentdir)
import time
import config
import models
from test_ic15 import to_bboxes
from dataset import get_img, scale


def order_points(pts):
    # print(" pajska",pts)
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    # print(" data" , warped, )
    # print(" point " , rect, )
    return warped, rect


class TextDetector:
    def __init__(self):
        self.model_path = config.model_path
        # self.img_path = ""
        if config.use_device:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        backbone = models.vgg16(pretrained=True, num_classes=18)
        for param in backbone.parameters():
            param.requires_grad = False
        model = backbone.to(self.device)
        if os.path.isfile(self.model_path):
            print(("Loading model and optimizer from checkpoint '{}'".format(self.model_path)))
            if config.use_device:
                checkpoint = torch.load(self.model_path)
            else:
                checkpoint = torch.load(self.model_path, map_location = self.device)
            # model.load_state_dict(checkpoint)
            d = collections.OrderedDict()
            for key, value in list(checkpoint['state_dict'].items()):
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print(("Loaded checkpoint '{}' (epoch {})"
                   .format(self.model_path, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format()))
            sys.stdout.flush()

        # model = model.eval()
        self.model = model

    def get_dataset(self, img_path, check_dewarp):

        # img_path = self.img_path
        long_size = config.test_long_size

        def get_item(long_size):
            img = get_img(img_path, check_dewarp)
            scaled_padded_image, scaled_img, _scale = scale(img, long_size)
            # cv2.imwrite("images-s.jpg",scaled_img)
            scaled_padded_image = Image.fromarray(scaled_padded_image)
            scaled_padded_image = scaled_padded_image.convert('RGB')
            scaled_padded_image = transforms.ToTensor()(scaled_padded_image)
            scaled_padded_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                scaled_padded_image)
            return img[:, :, [2, 1, 0]], scaled_padded_image, scaled_img, _scale

        org_img, scaled_padded_image, scaled_img, _scale = get_item(long_size)
        input_data = torch.utils.data.DataLoader(
            (org_img, scaled_padded_image),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=True)
        image = list(input_data)

        scaled_padded_image = image[1]
        org_img = image[0]
        return scaled_padded_image, org_img, scaled_img, _scale

    def predict(self, img_path, check_dewarp):
        sys.stdout.flush()
        image_name = img_path.split('/')[-1]
        # print("a",img_path)

        scaled_padded_image, org_img, scaled_img, _scale = self.get_dataset(img_path, check_dewarp)

        model = self.model
        model.eval()
        t1 = time.time()
        # print("**==**loaded model**==**")
        img = scaled_padded_image.to(self.device)
        # torch.cuda.synchronize(self.device)
        org_img = org_img.numpy().astype('uint8')[0]
        cls_logits, link_logits = model(img)
        outputs = torch.cat((cls_logits, link_logits), dim=1)
        shape = outputs.shape
        # print(org_img.shape)
        pixel_pos_scores = F.softmax(outputs[:, 0:2, :, :], dim=1)[:, 1, :, :]
        link_scores = outputs[:, 2:, :, :].view(shape[0], 2, 8, shape[2], shape[3])
        link_pos_scores = F.softmax(link_scores, dim=1)[:, 1, :, :, :]

        img_to_score_w_ratio = (scaled_padded_image.shape[1] / shape[3])
        img_to_score_h_ratio = (scaled_padded_image.shape[0] / shape[2])
        pixel_pos_scores = pixel_pos_scores[:, : int(scaled_img.shape[0] / img_to_score_h_ratio),
                           : int(scaled_img.shape[1] / img_to_score_w_ratio)]
        link_pos_scores = link_pos_scores[:, : int(scaled_img.shape[0] / img_to_score_h_ratio),
                          : int(scaled_img.shape[1] / img_to_score_w_ratio)]
        print("time start", time.time()-t1)
        t1 = time.time()
        mask, bboxes = to_bboxes(org_img, _scale, pixel_pos_scores.cpu().detach().numpy(),
                                 link_pos_scores.cpu().detach().numpy())                                               
        print("time t1", time.time()-t1)
        sys.stdout.flush()
        return org_img, bboxes, image_name

    def compute_prediction(self, org_img, bboxes, image_name):

        try:
            prediction = []
            # org_img, bboxes = self.predict(img_path)
            for b_idx, bbox in enumerate(bboxes):
                # print(bbox.reshape(4,2))
                # print(("B"))
                bbox = bbox.reshape(4, 2)
                cv2.drawContours(org_img, [bbox], -1, (0, 255, 0), 2)
                # img_crop,_ = self.crop_bbox(org_img, bbox, b_idx)
                # cv2.imwrite("media/detector/img.jpg", org_img)
                img_crop, box = four_point_transform(org_img, bbox)

                box = [int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])]
                if box[1] > box[3]:
                    a = box[1]
                    box[1]=box[3]
                    box[3] = a
                cv2.putText(org_img, str(b_idx+1), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                prediction.append([img_crop] + [box])
            cv2.imwrite("media/detector/bbox_{}".format(image_name), org_img)
            # print("exit")
            # exit()

        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction


    # def save_detection_to_xml(self,)
