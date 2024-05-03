import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
a = np.array([[1, 2], [2, 3], [1, 2]])
np.apply_along_axis(lambda k : np.array_equal(k, np.array([1, 2])), 1, a)

def is_box_color(pixel_color):
    b, g, r = pixel_color
    if b < 20 and g < 20 and r > 210:
        return True
    return False

def is_line_boundary(line):
    compare_func = lambda k : is_box_color(k)
    compare_res = np.apply_along_axis(compare_func, 1, line)
    equal_cnt = np.sum(compare_res)
    if equal_cnt > 30:
        return True
    return False

def get_2_edges(res):
    on_edge = False
    first_id = None
    for i in range(len(res)):
        if res[i]:
            on_edge = True
        else: 
            if on_edge:
                first_id = i
                break
    on_edge = False
    second_id = None
    for i in reversed(range(len(res))):
        if res[i]:
            on_edge = True
        else: 
            if on_edge:
                second_id = i
                break
    
    return first_id, second_id
    
def cropping_visual_box(image):
    pad = 5
    top, bottom, left, right = None, None, None, None
    check_boundary_func = lambda line : is_line_boundary(line)
    t2b_check_res = np.array([check_boundary_func(image[i, :, :].squeeze()) for i in range(image.shape[0])])
    top, bottom = get_2_edges(t2b_check_res)
    crop_img = None
    if top is not None and bottom is not None:
        top, bottom= top + pad, bottom - pad
        crop_img = image[top:bottom, :, :]
    
    if crop_img is not None and crop_img.shape[0] > 0:
        l2r_check_res = np.array([check_boundary_func(crop_img[:, i, :].squeeze()) for i in range(crop_img.shape[1])])
        left, right = get_2_edges(l2r_check_res)
    
    # image[top, :, :] = np.array([0, 0, 0])
    # image[bottom, :, :] = np.array([0, 0, 0])
    # image[:, left, :] = np.array([0, 0, 0])
    # image[:, right, :] = np.array([0, 0, 0])
    
    if left is not None and right is not None:
        left, right = left + pad, right - pad
        crop_img = crop_img[:, left:right, :]
    
    return crop_img

save_dir = "/home/asi/camera/thamnt/common/data_utils/cropping_from_vis/out"
for i, path in enumerate(tqdm(glob("/home/asi/camera/thamnt/dataset/data_vp/hoi_so/people_web/**/*.jpg", recursive=True))):
    if i < 13925:
        continue
    name = path.split("/")[-1]
    image = cv2.imread(path)
    box_color = np.array([75 , 79 , 74])
    crop_image = cropping_visual_box(image)
    if crop_image is not None and crop_image.shape[0] > 5 and crop_image.shape[1] > 5:
        cv2.imwrite(os.path.join(save_dir, name), crop_image)
    
    
