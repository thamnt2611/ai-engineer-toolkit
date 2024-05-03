import os
import matplotlib.pyplot as plt
import cv2
import collections
import numpy as np
from tqdm import tqdm

def plot_class_count(class_count_dict, out_path):
    plt.figure(figsize=(15, 5))
    loc_stat = collections.OrderedDict(sorted(loc_stat.items()))
    ax = plt.bar([LABELS[i] for i in loc_stat.keys()], loc_stat.values())
    plt.bar_label(ax)
    plt.savefig(out_path)

def plot_image(img, boxes, LABELS):
    w, h = img.shape[1], img.shape[0]
    for box in boxes: 
        cls_id, center_x, center_y, width, height = box[0], box[1]*w, box[2]*h, box[3]*w, box[4]*h
        cls_name = LABELS[int(cls_id)]
        if cls_name not in ['Police car',]:
            continue
        x1, y1, x2, y2 = center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, cls_name.upper(), (int(x1), int(y1 + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def random_vis_dataset(data_dir, vis_dir, number = 100):
    ex_ids = [i[:-4] for i in os.listdir(data_dir)]
    chosen_ids = np.random.choice(ex_ids, size=number)
    for ex_id in tqdm(chosen_ids):
        image_path = os.path.join(data_dir, ex_id + '.jpg')
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        img = cv2.imread(image_path)
        boxes = yolo_parse(label_path)
        # print(boxes[0])
        vis_img = plot_image(img, boxes)
        cv2.imwrite(os.path.join(vis_dir, ex_id + '.jpg'), vis_img)

def vis_image(data_dir, vis_dir):
    def ok(boxes):
        for box in boxes:
            cls_id, x, y, w, h = box
            cls_name = LABELS[int(cls_id)]
            if cls_name in ['Police car',]:
                return True
        return False
    ex_ids = [i[:-4] for i in os.listdir(data_dir)]
    for ex_id in tqdm(ex_ids):
        image_path = os.path.join(data_dir, ex_id + '.jpg')
        label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
        img = cv2.imread(image_path)
        boxes = yolo_parse(label_path)
        if ok(boxes):
            # print(boxes[0])
            vis_img = plot_image(img, boxes)
            cv2.imwrite(os.path.join(vis_dir, ex_id + '.jpg'), vis_img)

def plot_confustion_matrix():
    def plot(cm, normalize=True, save_dir='', names=(), on_plot=None):
    """
    Plot the confusion cm using seaborn and save it to a file.

    Args:
        normalize (bool): Whether to normalize the confusion cm.
        save_dir (str): Directory where the plot will be saved.
        names (tuple): Names of classes, used as labels on the plot.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
    """
    import seaborn as sn

    array = cm / ((cm.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc = len(names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = True  # apply names to ticklabels
    ticklabels = (list(names) + ['background']) if labels else 'auto'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty cm RuntimeWarning: All-NaN slice encountered
        sn.heatmap(array,
                    ax=ax,
                    annot=nc < 30,
                    annot_kws={
                        'size': 8},
                    cmap='Blues',
                    fmt='.2f' if normalize else '.0f',
                    square=True,
                    vmin=0.0,
                    xticklabels=ticklabels,
                    yticklabels=ticklabels).set_facecolor((1, 1, 1))
    print(array)
    title = 'Confusion cm' + ' Normalized' * normalize
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    plot_fname = save_dir
    fig.savefig(plot_fname, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(plot_fname)