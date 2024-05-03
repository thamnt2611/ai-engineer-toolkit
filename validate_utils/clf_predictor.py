
import sys

# sys.path.insert(0,'app/module/attribute/car_attr/car_color/yolov5_segment')

import os
import time
from PIL import Image
import torch
import numpy as np
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import disarray
import pandas as pd
# from .yolov5_segment.segment_car import MaskExtractor

path_cur=os.path.dirname(os.path.abspath(__file__))


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

class PlateColor(object):

    def __init__(self, model_path, classes):
        num_classes = len(classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.mask_extractor = MaskExtractor(self.device)
        # self.model_ft = models.resnet18(pretrained=False)
        # self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, num_classes)
        self.model_ft = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        self.model_ft.classifier = nn.Sequential(*[self.model_ft.classifier[0], self.model_ft.classifier[1], self.model_ft.classifier[2], nn.Linear(1024, num_classes)])
        self.model_ft.load_state_dict(torch.load( model_path, map_location = self.device))
        self.model_ft.eval()
        self.model_ft.to(self.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                         transforms.Resize((224,224)),
                         transforms.ToTensor(),
                         self.normalize])
        self.classes = classes
        self.param_color = []

    def predict(self, img):
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        inputs = self.transform(img).float()
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        pred=torch.nn.functional.softmax(outputs, dim=1)
        return self.classes[int(preds[0])], pred[0][int(preds[0])].item()

    def debug_vis(self, debug_dir, true_lb, pred_lb, img, image_name, prob, false_only = True):
        if false_only:
            if true_lb == pred_lb: 
                return
        if true_lb not in os.listdir(debug_dir):
            os.mkdir(os.path.join(debug_dir, true_lb))
        if pred_lb not in os.listdir(os.path.join(debug_dir, true_lb)):
            os.mkdir(os.path.join(debug_dir, true_lb, pred_lb))
        # cv2.putText(img, str(prob), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(debug_dir, true_lb, pred_lb, image_name), img)

    def validate(self, X, true_lbs, cm_save_path = 'cm.jpg'):
        preds = []
        for i, path in enumerate(tqdm(X)):
            img = cv2.imread(path)
            pred, prob = self.predict(img)
            preds.append(pred)
            image_name = path.split('/')[-1]
            # if true_lbs[i] != pred:
            #     print(prob)
            self.debug_vis('/home/asi/camera/thamnt/plate_color/debug_vis', true_lbs[i], pred, img, image_name, prob)
        cm = confusion_matrix(preds, true_lbs, labels=self.classes)
        plot(cm, False, cm_save_path , names=self.classes)
        df = pd.DataFrame(cm.T, index= self.classes, columns=self.classes)
        print(df.da.export_metrics().loc[['accuracy','f1', 'precision', 'recall']].to_csv())
        print(cm)

if __name__=='__main__':
    # classes = [i.lower() for i in ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Red', 'White', 'Yellow']]
    classes =  ['blue', 'red', 'white', 'yellow']
    model_path = '/home/asi/camera/thamnt/plate_color/cnn/models_v1/mobilenet/pc_mbnet_aug1_best.pt'
    predictor = PlateColor(model_path, classes)
    fdir = '/home/asi/camera/thamnt/dataset/plate_color/working_data/val'
    cm_save_path ='cm_mbnet'
    class_list = os.listdir(fdir)
    true_lbs = []
    X = []
    for c in class_list:
        if c.lower() in classes:
            image_names = os.listdir(os.path.join(fdir, c))
            images = [os.path.join(fdir, c, im) for im in image_names]
            images = [im for im in images if im is not None]
            X.extend(images)
            true_lbs.extend([c.lower()]*len(images))
    predictor.validate(X, true_lbs, cm_save_path)
