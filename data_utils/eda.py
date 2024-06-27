from lxml import etree
import os
import glob
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import shutil
from pathlib import Path
def yolo_parse(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().split('\n')
    if lines[-1] == '':
        lines = lines[:-1]
    boxes = []
    for line in lines:
        items = line.split(' ')
        box = [int(items[0])]
        box.extend([float(i) for i in items[1:]])
        boxes.append(box)
    return boxes

def plot_image(img, boxes):
    w, h = img.shape[1], img.shape[0]
    for box in boxes: 
        cls_id, center_x, center_y, width, height = box[0], box[1]*w, box[2]*h, box[3]*w, box[4]*h
        x1, y1, x2, y2 = center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, str(cls_id), (int(x1), int(y1 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img
import re
def random_vis_dataset(data_dir, vis_dir, number = 100):
    ex_ids = [i[:-4] for i in os.listdir(data_dir)]
    chosen_ids = np.random.choice(ex_ids, size=number)
    for ex_id in tqdm(chosen_ids):
        image_path = os.path.join(data_dir, ex_id + '.jpg')
        if not os.path.exists(image_path):
            image_path = image_path[:-4] + '.png'
        if not os.path.exists(image_path):
            continue
        # lb_ex_id = re.sub("unique", "", ex_id)
        # lb_ex_id = re.sub("[0-9]+_largest_dup", "", lb_ex_id)
        # while lb_ex_id.startswith("_"):
        #     lb_ex_id = lb_ex_id[1:]
        label_path = os.path.join("/home/asi/camera/thamnt/dataset/hho_det/hho_wp_clean_split/labels/train", ex_id + '.txt')
        print(image_path, label_path)
        img = cv2.imread(image_path)
        boxes = yolo_parse(label_path)
        print(len(boxes))
        vis_img = plot_image(img, boxes)
        cv2.imwrite(os.path.join(vis_dir, ex_id + '.jpg'), vis_img)

data_dir = '/home/asi/camera/thamnt/dataset/hho_det/hho_wp_clean_split/images/train'
vis_dir = '/home/asi/camera/thamnt/dataset/weapon_det/eda_data'
random_vis_dataset(data_dir, vis_dir)
