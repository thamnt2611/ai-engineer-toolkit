import glob
import shutil
import os
save_dir = "/home/asi/camera/thamnt/dataset/hho_vp_calibration_data"
for path in glob.glob("/home/asi/camera/thamnt/dataset/data_vp/**/*.jpg", recursive=True):
    name = path.split("/")[-1]
    if "per" in name:
        new_path = os.path.join(save_dir, name)
        shutil.copy(path, new_path)
