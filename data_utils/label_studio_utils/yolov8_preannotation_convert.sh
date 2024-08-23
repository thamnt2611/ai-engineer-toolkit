label-studio-converter import yolo -i /home/asi/dev/data_annotation/label-studio/data/data_upload/head_datasets/data_hat_nohat_det/data_hat -o hat_preannotated.json --image-root-url /data/local-files/?d=/label-studio/data/data_upload/head_datasets/head_det3/images
label-studio-converter import yolo -i /home/asi/camera/manhnd/deepstream/SODM/bank_security_cython/out/gendata/data -o manhnd_handhold.json --image-root-url /data/local-files/?d=/label-studio/data/data_upload/manhnd_handhold


sed 's#/home/asi/dev/data_annotation/label-studio/data/data_upload/head_det2/nohat/#/data/local-files/?d=/label-studio/data/data_upload/head_det2/nohat/images/#g' nohat_preannotated.json > fix_nohat_preannotate.json
