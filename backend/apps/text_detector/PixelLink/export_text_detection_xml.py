from text_detector import TextDetector, four_point_transform
import config 
import os
import requests

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import datetime
from xml.dom import minidom
text_detector = TextDetector()
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

images_path = requests.get("https://ai_products.gemvietnam.com/label/text-detection/list_files").json()
root_paths = [config.root_images_dir + x for x in images_path]
print(root_paths)
    
def export_prediction_text_detection():
    node_root = Element('dataset')

    node_name = SubElement(node_root, 'name')
    node_name.text = 'Tool Label OCR GEM'
    node_images = SubElement(node_root, 'images')

    for img_path in root_paths :
        # print(img_path)
        
        org_img, bboxes, image_name = text_detector.predict(img_path)
        org_h, org_w, _ = org_img.shape
        # print(image_name)
        
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
            print(height_box, ": ", width_box)
            node_box = SubElement(node_image, 'box', 
                {'height':str(height_box),
                'left':str(int(box[0][0])),
                'top':str(int(box[0][1])),
                'width': str(width_box)       
                }
            )
            node_label = SubElement(node_box, 'label')
            node_label.text = 'unlabelled'
    myfile = open(config.root_xml_dir+"writing_gt.xml", "w")
    myfile.write(prettify(node_root))
    # print(prettify(node_root))



#     return img_paths


if __name__ == '__main__':
    export_prediction_text_detection()