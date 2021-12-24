import os
import requests
import config
from dataset import get_bboxes, load_data_gt, get_img
from text_detector import four_point_transform

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import datetime
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
root_dir = "dataset/"
root_image_path = root_dir + "test_images/"
root_gt_path = root_dir + 'test_gt/'
data_dirs = [root_image_path]
gt_dirs = [root_gt_path]
img_paths, gt_paths = load_data_gt(data_dirs, gt_dirs)
print(len(img_paths), len(gt_paths))
# exit()
    
def export_gt_to_xml():
    node_root = Element('dataset')

    node_name = SubElement(node_root, 'name')
    node_name.text = 'Tool Label OCR GEM'
    node_images = SubElement(node_root, 'images')

    for idx, img_path in enumerate(img_paths) :
        print(img_path, "and", gt_paths[idx])
        org_img = get_img(img_path)
        image_name = img_path.split('/')[-1]
        
        bboxes, _ = get_bboxes(org_img, gt_paths[idx])
        org_h, org_w, _ = org_img.shape
        # print(bboxes)
        # exit()
        node_image = SubElement(node_images,'image', {'file':image_name} )

        node_width = SubElement(node_image, 'width')
        node_width.text = str(org_w)

        node_height = SubElement(node_image, 'height')
        node_height.text = str(org_h)

        for b_idx, bbox in enumerate(bboxes):
            bbox = bbox.reshape(4, 2)
            _, box = four_point_transform(org_img, bbox)
            height_box = int(box[2][1])-int(box[0][1])
            width_box = int(box[2][0])- int(box[0][0]) 
            # print(height_box, ": ", width_box)
            node_box = SubElement(node_image, 'box', 
                {'height':str(height_box),
                'left':str(int(box[0][0])),
                'top':str(int(box[0][1])),
                'width': str(width_box)       
                }
            )
            node_label = SubElement(node_box, 'label')
            node_label.text = 'unlabelled'
    myfile = open(config.root_xml_dir+"test_data_gt.xml", "w")
    myfile.write(prettify(node_root))
    # print(prettify(node_root))



#     return img_paths


if __name__ == '__main__':
    export_gt_to_xml()