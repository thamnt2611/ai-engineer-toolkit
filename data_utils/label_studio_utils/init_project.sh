label-studio init labeling-project --input-path=/Users/rahulsomani/Desktop/imgs/ --input-format=image-dir

# start

#!/bin/bash
cd /home/release/share/data_annotation/
source venv/bin/activate
export LOCAL_FILES_SERVING_ENABLED=1
nohup label-studio start -p 2222 --data-dir /home/release/share/data_annotation/label-studio/data > run.log &
