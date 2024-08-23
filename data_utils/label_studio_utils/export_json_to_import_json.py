import json
from pathlib import Path
import os 
import shutil
from tqdm import tqdm
json_path = "/home/release/share/data_annotation/label-studio/other_utils/parse_json_export/data/project-2-at-2024-06-12-04-34-7c020378.json"
with open(json_path, "r") as f:
    data = json.load(f)
    
new_data = []
save_dir = "/home/release/share/data_annotation/label-studio/data/data_upload/clean_shirt_color/part1"
for item in tqdm(data):
    path = item["image"].replace("/data/local-files/?d=", "/home/release/share/data_annotation/")
    if "shirt-color" in item.keys():
        name = path.split("/")[-1]
        new_path = os.path.join(save_dir, name)
        new_item = {"data": {"image": "/data/local-files/?d=" + new_path}}
        annot_res = item["shirt-color"] 
        annot_res = annot_res["choices"] if isinstance(annot_res, dict) else [annot_res]
        new_item["predictions"] = [
            {
                "model_version": "one",
                "score": 0.5,
                "result": [
                    {
                        "type": "choices",
                        "from_name": "shirt-color", "to_name": "image",
                        "value": {
                            "choices": annot_res
                        }
                    }
                ]
            }
        ]
        new_data.append(new_item)
        # shutil.copy(path, new_item["image"])

with open("/home/release/share/data_annotation/label-studio/other_utils/parse_json_export/data/preannotated_part1.json",
          "w") as f:
    json.dump(new_data, f)
    
        
    
        
