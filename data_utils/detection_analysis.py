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
LABELS = [] #FIXME
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

def yolo_parse(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
    boxes = []
    for line in lines:
        items = line.split(' ')
        box = [int(items[0])]
        box.extend([float(i) for i in items[1:]])
        boxes.append(box)
    return boxes

def box_xyxy2box_nxywh(box, img_w, img_h):
    label, xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return [label, center_x / img_w, center_y / img_h, width / img_w, height / img_h]

def dump_boxes_to_file(boxes, filepath):
    yolo_label_str = ''
    for yolo_box in boxes:
        cls_id, x_center, y_center, width, height = yolo_box
        yolo_label_str += ' '.join([str(cls_id), str(x_center), str(y_center), str(width), str(height)]) +'\n'
    with open(filepath, 'w') as f:
        f.write(yolo_label_str)


def convert_voc2yolo(voc_dirs, yolo_dir, lb2idx):
    print("convert_voc2yolo")
    ex_id_set = set()
    i = 0
    Path(os.path.join(yolo_dir), 'images').mkdir(parents=True, exist_ok=True)
    Path(os.path.join(yolo_dir), 'labels').mkdir(parents=True, exist_ok=True)
    for dir in voc_dirs:
        image_dir = os.path.join(dir, 'images')
        label_dir = os.path.join(dir, 'annotations')
        for image_path in tqdm(glob.glob(image_dir + "/**/*.png", recursive=True)):
            i+= 1
            ex_id = image_path.split("/")[-1][:-4]
            label_path = os.path.join(label_dir, ex_id + '.xml')
            # location = [i for i in LOCATIONS if i in image_path][0]
            assert ex_id not in ex_id_set
            ex_id_set.add(ex_id)
            if not os.path.exists(label_path):
                with open('voc2yolo_debug.txt', 'a') as f:
                    f.write(image_path + '\n')
                continue
            img = cv2.imread(image_path)
            img_w, img_h = img.shape[1], img.shape[0]
            boxes = voc_parse(label_path)
            yolo_label_str = ''
            for box in boxes:
                yolo_box = box_xyxy2box_nxywh(box, img_w, img_h)
                cls_name, x_center, y_center, width, height = yolo_box
                cls_id = lb2idx[cls_name]
                yolo_label_str += ' '.join([str(cls_id), str(x_center), str(y_center), str(width), str(height)]) +'\n'
            new_ex_id = ex_id
            new_img_location = os.path.join(yolo_dir, 'images', new_ex_id + '.jpg')
            yolo_label_location = os.path.join(yolo_dir, 'labels', new_ex_id + '.txt')
            cv2.imwrite(new_img_location, img)
            with open(yolo_label_location, 'w') as f:
                f.write(yolo_label_str)

def get_class_count_voc(annotation_dirs):
    class_count = {}
    for dir in annotation_dirs:
        for label_path in tqdm(glob.glob(dir + '/*')):
            boxes = voc_parse(label_path)
            for box in boxes:
                class_name, _, _, _, _ = box
                if class_name in class_count.keys():
                    class_count[class_name] += 1
                else:
                    class_count[class_name] = 1
    print(class_count)

def get_class_count_yolo(annotation_dir):
    class_list = []
    for label_path in tqdm(glob.glob(annotation_dir + '/*')):
        boxes = yolo_parse(label_path)
        class_list.extend([box[0] for box in boxes])
    from collections import Counter
    stat = Counter(class_list)
    res = dict()
    for cls_id in stat.keys():
        res[LABEL[cls_id]] = stat[cls_id]
    return res

def train_val_split_by_filtering(meta):
    import shutil
    def loc_id(ex_id):
        loc = ex_id.split('_')[0]
        cam = meta[ex_id]['cam_id']
        return loc + '_' + cam
    train_ids = []
    val_ids = []
    data_dir = '/home/asi/camera/thamnt/dataset/vehicle_detect_0/yolo_format'
    for ex_id, _ in meta.items():
        if loc_id(ex_id) in ['hcm_4', 'hy_15']:
            val_ids.append(ex_id)
            old_dir = os.path.join(data_dir, 'images', ex_id + '.jpg')
            new_dir = os.path.join(data_dir, 'images', 'val', ex_id + '.jpg')
            shutil.move(old_dir, new_dir)

            old_dir = os.path.join(data_dir, 'labels', ex_id + '.txt')
            new_dir = os.path.join(data_dir, 'labels', 'val', ex_id + '.txt')
            shutil.move(old_dir, new_dir)

        else:
            train_ids.append(ex_id)
            old_dir = os.path.join(data_dir, 'images', ex_id + '.jpg')
            new_dir = os.path.join(data_dir, 'images', 'train', ex_id + '.jpg')
            shutil.move(old_dir, new_dir)

            old_dir = os.path.join(data_dir, 'labels', ex_id + '.txt')
            new_dir = os.path.join(data_dir, 'labels', 'train', ex_id + '.txt')
            shutil.move(old_dir, new_dir)

def aggregate_and_split(data_dirs, save_dir, split_ratio=0.8):
    seed = 11
    image_paths = []
    for dir in data_dirs:
        for image_path in tqdm(glob(dir + '/images/*')):
            ex_id = image_path.split('/')[-1][:-4]
            image_paths.append(image_path)
    random.Random(seed).shuffle(image_paths)
    bdr = int(0.8 * len(image_paths)) + 1
    train_names, val_names = image_paths[:bdr], image_paths[bdr:]
    for n in tqdm(train_names):
        img = cv2.imread(n)
        ex_id = n.split('/')[-1][:-4]
        new_dir = os.path.join(save_dir, 'images', 'train', ex_id + '.jpg')
        cv2.imwrite(new_dir, img)
        old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
        new_dir = os.path.join(save_dir, 'labels', 'train', ex_id + '.txt')
        shutil.copy(old_dir, new_dir)
    for n in tqdm(val_names):
        img = cv2.imread(n)
        ex_id = n.split('/')[-1][:-4]
        new_dir = os.path.join(save_dir, 'images', 'val', ex_id + '.jpg')
        cv2.imwrite(new_dir, img)

        old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
        new_dir = os.path.join(save_dir, 'labels', 'val', ex_id + '.txt')
        shutil.copy(old_dir, new_dir)

# def aggregate_and_split(data_dirs, save_dir, split_ratio=0.8):
#     seed = 11
#     attr2idxs = {}
#     for dir in data_dirs:
#         for image_path in tqdm(glob.glob(dir + '/images/*')):
#             ex_id = image_path.split('/')[-1][:-4]
#             location = ex_id.split('_')[0]
#             if location == 'tn':
#                 cam_id = ex_id.split('_')[-1]
#             else:
#                 cam_id = ex_id.split('_')[2]
#             attr = location + '_' + cam_id
#             if attr not in attr2idxs.keys():
#                 attr2idxs[attr] = [image_path]
#             else:
#                 attr2idxs[attr].append(image_path)
#     print(attr2idxs.keys())
#     for attr, image_paths in attr2idxs.items():
#         random.Random(seed).shuffle(image_paths)
#         bdr = int(0.8 * len(image_paths)) + 1
#         train_names, val_names = image_paths[:bdr], image_paths[bdr:]
#         for n in tqdm(train_names):
#             img = cv2.imread(n)
#             ex_id = n.split('/')[-1][:-4]
#             new_dir = os.path.join(save_dir, 'images', 'train', ex_id + '.jpg')
#             cv2.imwrite(new_dir, img)
#             old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
#             new_dir = os.path.join(save_dir, 'labels', 'train', ex_id + '.txt')
#             shutil.copy(old_dir, new_dir)
#         for n in tqdm(val_names):
#             img = cv2.imread(n)
#             ex_id = n.split('/')[-1][:-4]
#             new_dir = os.path.join(save_dir, 'images', 'val', ex_id + '.jpg')
#             cv2.imwrite(new_dir, img)

#             old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
#             new_dir = os.path.join(save_dir, 'labels', 'val', ex_id + '.txt')
#             shutil.copy(old_dir, new_dir)


def plot_class_count(class_count_dict, out_path):
    plt.figure(figsize=(15, 5))
    loc_stat = collections.OrderedDict(sorted(loc_stat.items()))
    ax = plt.bar([LABELS[i] for i in loc_stat.keys()], loc_stat.values())
    plt.bar_label(ax)
    plt.savefig(out_path)

def plot_image(img, boxes):
    w, h = img.shape[1], img.shape[0]
    for box in boxes: 
        cls_id, center_x, center_y, width, height = box[0], box[1]*w, box[2]*h, box[3]*w, box[4]*h
        x1, y1, x2, y2 = center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        # cv2.putText(img, LABELS[int(cls_id)].upper(), (int(x1), int(y1 + 10)),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def random_vis_dataset(data_dir, vis_dir, number = 100):
    ex_ids = [i[:-4] for i in os.listdir(data_dir)]
    chosen_ids = np.random.choice(ex_ids, size=number)
    for ex_id in tqdm(chosen_ids):
        image_path = os.path.join(data_dir, ex_id + '.jpg')
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        print(image_path, label_path)
        img = cv2.imread(image_path)
        boxes = yolo_parse(label_path)
        # print(boxes[0])
        vis_img = plot_image(img, boxes)
        cv2.imwrite(os.path.join(vis_dir, ex_id + '.jpg'), vis_img)

if __name__ == '__main__':
    # annotation_dirs = ['/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_98/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_99/annotations',
    #                 #    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_100/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_101/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_102/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_103/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_104/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_105/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_106/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_107/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_108/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_109/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_110/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_111/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_112/annotations',
    #                    '/home/asi/camera/thamnt/dataset/vehicle_detect/tn_20230626/Data_TN_26_06_2023/camera_113/annotations',
    #                    ]
    # # get_class_count_voc(annotation_dirs)

    # for dir in annotation_dirs:
    #     ann_paths = [os.path.join(dir, name) for name in os.listdir(dir)]
    #     for path in tqdm(ann_paths):
    #         boxes = voc_parse(path)
    #         par_boxes = []
    #         for box in boxes:
    #             cls_box = box[0]
    #             xmin, ymin, xmax, ymax = box[1:]
    #             if cls_box in ['no_helmet', 'hat', 'helmet']:
    #                 for box2 in boxes:
    #                     cls_box2, xmin2, ymin2, xmax2, ymax2 = box2
    #                     if xmin2 < xmin and ymin2 < ymin and xmax2 > xmax and ymax2 > ymax:
    #                         par_box = list(box2) + [cls_box]
    #                         par_boxes.append(par_box)
    #         image_path = path.replace('annotations', 'images')[:-4] + '.jpg'
    #         image = cv2.imread(image_path)
    #         id = path.split('/')[-1][:-4]
    #         for man_box in par_boxes:
    #             xmin, ymin, xmax, ymax = man_box[1:-1]
    #             man_class = man_box[0]
    #             hat_class = man_box[-1]
    #             man_image = image[ymin:ymax, xmin:xmax]
    #             name = hat_class + '_' + id + '_' + man_class + '.jpg'
    #             save_path = os.path.join('/home/asi/camera/thamnt/dataset/hat_dataset/tn_filter', name)
    #             cv2.imwrite(save_path, man_image)

    # data_dir = '/home/asi/camera/thamnt/dataset/smartphone_det/per_crop_filter/images'
    # vis_dir = '/home/asi/camera/thamnt/dataset/smartphone_det/debug'
    # random_vis_dataset(data_dir, vis_dir, number = 100)
    
    
    voc_dirs = ["/home/asi/camera/thamnt/dataset/weapon_det/asi_weapon_det_duicui_voc"]
    yolo_dir = "/home/asi/camera/thamnt/dataset/weapon_det/asi_weapon_det_duicui_yolo"
    convert_voc2yolo(voc_dirs, yolo_dir, {"nguoi": 0, "dui cui": 1})


        