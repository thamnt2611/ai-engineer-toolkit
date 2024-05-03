def aggregate_and_split(data_dirs, save_dir, split_ratio=0.8):
    seed = 11
    attr2idxs = {}
    for dir in data_dirs:
        for image_path in tqdm(glob.glob(dir + '/images/*')):
            ex_id = image_path.split('/')[-1][:-4]
            location = ex_id.split('_')[0]
            if location == 'tn':
                cam_id = ex_id.split('_')[-1]
            else:
                cam_id = ex_id.split('_')[2]
            attr = location + '_' + cam_id
            if attr not in attr2idxs.keys():
                attr2idxs[attr] = [image_path]
            else:
                attr2idxs[attr].append(image_path)
    print(attr2idxs.keys())
    for attr, image_paths in attr2idxs.items():
        random.Random(seed).shuffle(image_paths)
        bdr = int(0.8 * len(image_paths)) + 1
        train_names, val_names = image_paths[:bdr], image_paths[bdr:]
        for n in tqdm(train_names):
            img = cv2.imread(n)
            ex_id = n.split('/')[-1][:-4]
            new_dir = os.path.join(save_dir, 'images', 'train', ex_id + '.jpg')
            cv2.imwrite(new_dir, img)
            old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
            new_dir = os.path.join(save_dir, 'labels', 'train', ex_id + '.txt')
            shutil.copy(old_dir, new_dir)
        for n in tqdm(val_names):
            img = cv2.imread(n)
            ex_id = n.split('/')[-1][:-4]
            new_dir = os.path.join(save_dir, 'images', 'val', ex_id + '.jpg')
            cv2.imwrite(new_dir, img)

            old_dir = n.replace('images', 'labels').replace('jpg', 'txt')
            new_dir = os.path.join(save_dir, 'labels', 'val', ex_id + '.txt')
            shutil.copy(old_dir, new_dir)

def train_val_split_by_filtering(meta):
    def get_all_example_metadata(annotation_dirs):
        meta_dict = {}
        for dir in annotation_dirs:
            for label_path in tqdm(glob.glob(dir + '/*')):
                ex_id = label_path.split("/")[-1][:-4]
                boxes = yolo_parse(label_path)
                obj_list = [box[0] for box in boxes]
                cam_id = get_camid_from_exid(ex_id)
                if ex_id in meta_dict.keys():
                    print(label_path)
                    print(label_path.replace('val', 'train'))
                    print()
                    continue
                assert ex_id not in meta_dict.keys()
                meta_dict[ex_id] = {
                    'cam_id': cam_id,
                    'obj_list': obj_list
                }
        return meta_dict
    import shutil
    def loc_id(ex_id):
        loc = ex_id.split('_')[0]
        cam = meta[ex_id]['cam_id']
        return loc + '_' + cam
    train_ids = []
    val_ids = []
    data_dir = '/home/asi/camera/thamnt/dataset/vehicle_detect_0/yolo_format'
    for ex_id, _ in meta.items():
        if loc_id(ex_id) in ['hcm_4', 'hy_15']:
            val_ids.append(ex_id)
            old_dir = os.path.join(data_dir, 'images', ex_id + '.jpg')
            new_dir = os.path.join(data_dir, 'images', 'val', ex_id + '.jpg')
            shutil.move(old_dir, new_dir)

            old_dir = os.path.join(data_dir, 'labels', ex_id + '.txt')
            new_dir = os.path.join(data_dir, 'labels', 'val', ex_id + '.txt')
            shutil.move(old_dir, new_dir)

        else:
            train_ids.append(ex_id)
            old_dir = os.path.join(data_dir, 'images', ex_id + '.jpg')
            new_dir = os.path.join(data_dir, 'images', 'train', ex_id + '.jpg')
            shutil.move(old_dir, new_dir)

            old_dir = os.path.join(data_dir, 'labels', ex_id + '.txt')
            new_dir = os.path.join(data_dir, 'labels', 'train', ex_id + '.txt')
            shutil.move(old_dir, new_dir)
