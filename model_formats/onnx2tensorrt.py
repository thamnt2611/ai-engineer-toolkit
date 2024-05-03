import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
# More: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-developer-guide/index.html#python_topics
# Step 1: Importing TensorRT Into Python
import tensorrt as trt
 
## logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

# Step 2: Creating A Network Definition In Python (https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-developer-guide/index.html#network_python)
# Note: Since the ONNX format is quickly developing, you may encounter a version mismatch between the model version and the parser version. 
# Best practice: In general, the newer version of the OnnxParser is designed to be backward compatible
# initialize TensorRT engine 
def build_engine_onnx(onnx_file_path, tensorrt_engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    config.set_flag(trt.BuilderFlag.FP16)

    # parse ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    inputTensor = network.get_input(0) 
    print('inputTensor.name:', inputTensor.name)
    dynamic_axes = True
    min_engine_batch_size = 1 # min input data size of model for inference
    opt_engine_batch_size = 8 # most used input data size of model for inference
    max_engine_batch_size = 16 # max input data size of model for inference
    img_size = [3, 224, 224]
    if dynamic_axes:
        profile.set_shape(inputTensor.name, (min_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(opt_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
        print('Set dynamic')
    else:
        batch_size = 8
        profile.set_shape(inputTensor.name, (batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2])) 
    config.add_optimization_profile(profile)

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)
onnx_file_path = '/home/asi/camera/thamnt/car_logo/cl_res18_29cls_org_best.onnx'
tensorrt_engine_path = '/home/asi/camera/thamnt/car_logo/cl_res18_29cls_org_best.engine'
build_engine_onnx(onnx_file_path, tensorrt_engine_path)
