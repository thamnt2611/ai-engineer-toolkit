# label-studio export 32 JSON --data-dir /home/asi/dev/data_annotation/label-studio/data #NOT WORK

label-studio shell --data-dir /home/asi/dev/data_annotation/label-studio/data
# inside shell

import json
from tasks.serializers import TaskWithAnnotationsSerializer

tasks = Task.objects.filter(project=32)
export_json_data = TaskWithAnnotationsSerializer(tasks, many=True).data
with open('output.json', 'w') as f:
    json.dump(export_json_data, f)




sed 's#/data/local-files/?d=/label-studio/data/data_upload/handhold_face_filter/images/#/home/asi/dev/data_annotation/label-studio/data/data_upload/handhold_face_filter/images/#g' hho_clean_all.json > fixed_hho_clean_all.json


label-studio-converter export -i fixed_hho_clean_all.json -f YOLO -o /home/asi/dev/data_annotation/public_datasets/hand_object_detector_dataset/hho_face_filter_yolo -c /home/asi/dev/data_annotation/public_datasets/hand_object_detector_dataset/preannotated.label_config.xml
