import os

from ultralytics import YOLO
import cv2
import glob
from tqdm import tqdm
LABELS = [
    'board'
]
model =YOLO("/home/asi/camera/thamnt/models/yolov8/meter_board_det/yolo_gas_detection.pt")
image_dir = '/home/asi/camera/thamnt/dataset/gas_board_det/images'
output_dir = '/home/asi/camera/thamnt/dataset/gas_board_det/labels'
for image_path in tqdm(glob.glob(image_dir + '/*')):
    print(image_path)
    image_id = image_path.split('/')[-1][:-4]
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    results = model(img)[0]
    print(results)
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(img, LABELS[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, image_id + ".jpg"), img)


        #TODO