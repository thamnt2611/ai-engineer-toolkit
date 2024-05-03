import cv2
import json
import os
from tqdm import tqdm
def create_item(image_path, label):
  item = {
    "data": {
      "image": '/data/local-files/?d=' + image_path.replace('/home/asi/dev/data_annotation', '')
    },
    "predictions": [{
      "model_version": "raw",
      "score": None,
      "result": [
        {
          "id": "result_choice",
          "type": "choices",
          "from_name": "location", "to_name": "image",
          "value": {
            "choices": ["Outside"]
          }
        }]
    }]
  }
#   for i, box in enumerate(boxes):
#       cls_id, x, y, w, h = box
#       label = "Fire" if cls_id == 1 else "Smoke"
#       res_item = {
#           "id": f"result_box{i}",
#           "type": "rectanglelabels",        
#           "from_name": "fire_box", "to_name": "image",
#           "original_width": image_w, "original_height": image_h,
#           "image_rotation": 0,
#           "value": {
#             "rotation": 0,          
#             "x": (x - w/2) * 100, "y": (y - h/2) * 100,
#             "width": w * 100, "height": h * 100,
#             "rectanglelabels": [label]
#           }
#       }
#       item["predictions"][0]["result"].append(res_item)
  return item

v8_dir = '/home/asi/dev/data_annotation/label-studio/data/data_upload/fire_dataset/yolov8_preannotated/1_cleaned_det'
datasets = os.listdir(v8_dir)
cnt = 0
import_dict = []
for dataset in datasets:
    image_names = os.listdir(os.path.join(v8_dir, dataset, 'images'))
    for name in tqdm(image_names):
        image_path = os.path.join(v8_dir, dataset, 'images', name)
        label_path = os.path.join(v8_dir, dataset, 'labels', name[:-4] + '.txt')
        boxes = yolo_parse(label_path)
        image = cv2.imread(image_path)
        image_w, image_h = image.shape[1], image.shape[0]
        data_item = create_item(image_path, boxes, image_w, image_h)
        import_dict.append(data_item)
with open(f'/home/asi/dev/data_annotation/label-studio/data/data_upload/fire_dataset/yolov8_preannotated/1_cleaned_det_preannotated_outside_only.json', 'w') as f:
    json.dump(import_dict, f)
