import os
import pandas as pd
import random
import cv2
from torchvision import datasets, models, transforms
def get_clscount_folder(fdir):
    cls_count = {}
    class_list = os.listdir(fdir)
    for c in class_list:
        images = os.listdir(os.path.join(fdir, c))
        cls_count[c] = len(images)

    return cls_count

def aggregate_and_split(data_dirs, LABELS, split_ratio=0.8):
    seed = 11
    for cls_name in LABELS:
        for src_name, ddir in data_dirs.items():
            sub_folders = [n.lower() for n in os.listdir(ddir)]
            if cls_name in sub_folders:
                print(cls_name)
                image_names = os.listdir(os.path.join(ddir, cls_name))
                random.Random(seed).shuffle(image_names)
                bdr = int(0.8 * len(image_names)) + 1
                train_names, val_names = image_names[:bdr], image_names[bdr:]
                for n in train_names:
                    img = cv2.imread(os.path.join(ddir, cls_name, n))
                    train_dir = '/home/asi/camera/thamnt/dataset/plate_color/working_data/train'
                    if cls_name not in os.listdir(train_dir):
                        os.mkdir(os.path.join(train_dir, cls_name))
                    cv2.imwrite(os.path.join(train_dir, cls_name, src_name + '_' + n), img)
                for n in val_names:
                    img = cv2.imread(os.path.join(ddir, cls_name, n))
                    val_dir = '/home/asi/camera/thamnt/dataset/plate_color/working_data/val'
                    if cls_name not in os.listdir(val_dir):
                        os.mkdir(os.path.join(val_dir, cls_name))
                    cv2.imwrite(os.path.join(val_dir, cls_name, src_name + '_' + n), img)

if __name__=='__main__':
    

    datadirs = {
        'our': '/home/asi/camera/thamnt/dataset/plate_color/lp_color',
        'ex': '/home/asi/camera/thamnt/dataset/plate_color/our_cropped'
    }
    LABELS = ['blue', 'red', 'white', 'yellow']
    aggregate_and_split(datadirs, LABELS, split_ratio=0.8)

    dir = '/home/asi/camera/thamnt/dataset/plate_color/working_data/train'
    cls_count = get_clscount_folder(dir)
    df = pd.DataFrame(cls_count, index=[0])
    print(dir)
    print(df.to_csv())

    dir = '/home/asi/camera/thamnt/dataset/plate_color/working_data/val'
    cls_count = get_clscount_folder(dir)
    df = pd.DataFrame(cls_count, index=[0])
    print(dir)
    print(df.to_csv())

