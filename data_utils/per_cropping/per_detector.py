import ultralytics
import cv2
from glob import glob
import os
from tqdm import tqdm
from lxml import etree
model = ultralytics.YOLO("/home/asi/camera/thamnt/common/data_utils/per_cropping/model/v8n_2cls_best.pt")

with open('/home/asi/camera/thamnt/common/data_utils/per_cropping/per_box_test.txt', 'w'):
    pass

def voc_parse(xml_path):
    boxes = []
    doc = etree.parse(xml_path)

    for obj in doc.xpath('//object'):
        name = obj.xpath('./name/text()')[0]
        xmin = int(float(obj.xpath('./bndbox/xmin/text()')[0]))
        ymin = int(float(obj.xpath('./bndbox/ymin/text()')[0]))
        xmax = int(float(obj.xpath('./bndbox/xmax/text()')[0]))
        ymax = int(float(obj.xpath('./bndbox/ymax/text()')[0]))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes

def is_ok(boxes):
    for box in boxes:
        cls_name = box[0]
        if cls_name == 'smartphone':
            return True
    return False

for img_path in tqdm(glob("/home/asi/camera/thamnt/dataset/weapon_det/train3/images/*.jpg")):
    img_name = img_path.split('/')[-1][:-4]
    img = cv2.imread(img_path)
    results = model(img_path)[0]
    for i, result in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = result
        
        per_img = img[int(y1):int(y2), int(x1):int(x2)]
        if per_img.shape[0] > 40 and per_img.shape[1] > 40 and score > 0.6:
            # box_str = f'{img_name}_{i} {x1} {y1} {x2} {y2} {score}\n'
            # with open('/home/asi/camera/thamnt/common/data_utils/per_cropping/per_box_test.txt', 'a') as f:
            #     f.write(box_str)
            per_dir = f"/home/asi/camera/thamnt/common/data_utils/per_cropping/per_cropping_train3/{img_name}_{i}.jpg"
            cv2.imwrite(per_dir, per_img)
