import os
import json
from glob import glob
image_path = "/home/asi/camera/thamnt/common/data_utils/cropping_from_vis/out_done"
save_path = "/home/asi/camera/thamnt/common/data_utils/cropping_from_vis/out_done.json"
all_data = []
for path in glob(f"{image_path}/*.jpg"):
    item = {"data": {"image": f"/data/local-files/?d={path}"}}
    all_data.append(item)

print(all_data)
with open(save_path, "w") as f:
    json.dump(all_data, f)
