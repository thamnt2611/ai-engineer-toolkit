import json
from pathlib import Path
import os 
import shutil
from tqdm import tqdm
json_path = "/home/release/share/data_annotation/label-studio/data/export/project-52-at-2024-06-27-06-57-a8c7d5b4.json"
with open(json_path, "r") as f:
    data = json.load(f)
save_dir = "/home/release/share/data_annotation/label-studio/other_utils/car_color_utils/data/car_color_v1"
for item in tqdm(data):
    path = item["data"]["image"].replace("/data/local-files/?d=", "")
    try:
        anno = item["annotations"][0]["result"][0]["value"]
        color = "-".join(anno["choices"]).replace(" ", "").lower().replace("/","").replace("car-", "")
        # if "shirt-color" in item.keys():
            # if isinstance(item["shirt-color"], dict):
            #     color = "-".join(item["shirt-color"]["choices"]).replace(" ", "").lower().replace("/","-").replace("shirt-", "")
            # else:
            #     color = item["shirt-color"].strip().lower().replace("/","-").replace("shirt-", "")
        name = path.split("/")[-1]
        shirt_dir = os.path.join(save_dir, color)
        # print(type(color))
        Path(shirt_dir).mkdir(exist_ok=True, parents=True)
        shutil.copy(path, os.path.join(shirt_dir, name))
    except: 
        continue
        
        
