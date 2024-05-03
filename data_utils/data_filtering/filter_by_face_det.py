from glob import glob
from ultralytics import YOLO
import shutil
import os
from tqdm import tqdm 

def has_target_object(img_path, model, target_id):
    results = model.predict(img_path, save=False, device="cuda:2")[0]
    
    for i, result in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = result
        if class_id == target_id and score > 0.1:
            return True
    return False

    
def fileter_by_target_object_and_save(image_folder, model, target_id, have_save_dir, nohave_save_dir):
    result_paths = []
    for path in tqdm(glob(f"{image_folder}/*")):
        name = path.split("/")[-1]
        if has_target_object(path, model, target_id):
            result_paths.append(path)
            new_path = os.path.join(have_save_dir, name)
            shutil.copy(path, new_path)
        else:
            new_path = os.path.join(nohave_save_dir, name)
            shutil.copy(path, new_path)
            # lb_path = path.replace("/images/", "/labels/")[:-4] + '.txt'
            # new_lbpath = new_path.replace("/images/", "/labels/")[:-4] + '.txt'
            # shutil.copy(lb_path, new_lbpath)
    return result_paths

if __name__ == "__main__":
    image_folder = "/home/asi/camera/thamnt/dataset/weapon_det/weapon_crop/cleaned_face_filter/images"
    model_path = "/home/asi/camera/thamnt/common/det_models/yolov8m-face.pt"
    have_save_dir = "/home/asi/camera/thamnt/dataset/weapon_det/weapon_crop/cleaned_face_filter/have_face"
    nohave_save_dir = "/home/asi/camera/thamnt/dataset/weapon_det/weapon_crop/cleaned_face_filter/no_face"
    model = YOLO(model_path)
    target_id = 0
    filter_paths = fileter_by_target_object_and_save(image_folder, model, target_id, have_save_dir, nohave_save_dir)
    # for path in filter_paths:
    #     name = path.split("/")[-1]
    #     new_path = os.path.join(save_dir, name)
    #     shutil.copy(path, new_path)
        
    
            
        
