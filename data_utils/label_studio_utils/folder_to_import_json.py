import json
from pathlib import Path
import os 
import shutil
from tqdm import tqdm
from glob import glob


def folder2cls(name):
    name = "Car-" + name[0].upper() + name[1:]
    return name 
def folder2json_cls(folder):
    data = []
    subfolders = os.listdir(folder)
    for sub in subfolders:
        clsname = folder2cls(sub)
        for path in glob(folder + "/" + sub + "/*.jpg"):
            new_item = {"data": {"image": "/data/local-files/?d=" + path}}
            new_item["predictions"] = [
                {
                    "model_version": "one",
                    "score": 0.5,
                    "result": [
                        {
                            "type": "choices",
                            "from_name": "car-color", "to_name": "image",
                            "value": {
                                "choices": [clsname]
                            }
                        }
                    ]
                }
            ]
            data.append(new_item)
    return data

data = folder2json_cls("/home/release/share/data_annotation/label-studio/data/data_upload/car_color/car_color_filter_sub12")
with open("/home/release/share/data_annotation/label-studio/data/data_upload/car_color/car_color_filter_sub12_import.json",
          "w") as f:
    json.dump(data, f)
    
        
    
        
