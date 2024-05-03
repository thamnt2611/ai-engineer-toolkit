def box_xyxy2box_nxywh(box, img_w, img_h):
    label, xmin, ymin, xmax, ymax = box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return [label, center_x / img_w, center_y / img_h, width / img_w, height / img_h]

def box_nxywh2xyxy(box, img_w, img_h):
    cls_id, center_x, center_y, width, height = box[0], box[1]*img_w, box[2]*img_h, box[3]*img_w, box[4]*img_h
    x1, y1, x2, y2 = center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2
    return [cls_id, x1, y1, x2, y2]