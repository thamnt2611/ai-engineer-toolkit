def dump_box_to_txt_file(boxes, txt_path):
    label_str = ''
    for box in boxes:
        cls_id, x_center, y_center, width, height = box
        label_str += ' '.join([str(cls_id), str(x_center), str(y_center), str(width), str(height)]) +'\n'
    with open(txt_path, 'w') as f:
        f.write(label_str)

