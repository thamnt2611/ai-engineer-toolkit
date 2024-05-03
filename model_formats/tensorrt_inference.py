import argparse
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from collections import OrderedDict
from PIL import Image
import cv2
import time

# Source: https://github.com/qbxlvnf11/convert-pytorch-onnx-tensorrt/blob/TensorRT-21.08/convert_onnx_to_tensorrt/convert_onnx_to_tensorrt.py#L160 
# Torch
import torch
from torch import nn
import torchvision.models as models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# ONNX: pip install onnx, onnxruntime
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

# CUDA & TensorRT
import pycuda.driver as cuda 
# from pycuda.driver import cuda 
import pycuda.autoinit
import tensorrt as trt
from tqdm import tqdm

TRT_LOGGER = trt.Logger()

def string_to_bool(args):

    if args.dynamic_axes.lower() in ('true'): args.dynamic_axes = True
    else: args.dynamic_axes = False

    return args

def get_transform(img_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        normalize])
    return transform

def build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, dynamic_axes, \
	img_size, batch_size, min_engine_batch_size, opt_engine_batch_size, max_engine_batch_size):
    
    # Builder
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
            	print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    
    # Input
    inputTensor = network.get_input(0) 
    # Dynamic batch (min, opt, max)
    print('inputTensor.name:', inputTensor.name)
    if dynamic_axes:
        profile.set_shape(inputTensor.name, (min_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(opt_engine_batch_size, img_size[0], img_size[1], img_size[2]), \
        	(max_engine_batch_size, img_size[0], img_size[1], img_size[2]))
        print('Set dynamic')
    else:
        profile.set_shape(inputTensor.name, (batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2]), \
        	(batch_size, img_size[0], img_size[1], img_size[2]))
    config.add_optimization_profile(profile)
    #network.unmark_output(network.get_output(0))
    
    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)
        
def trt_inference(engine, context, data):  
    # Allocate host and device buffers
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        print("size", size, binding_idx)
        dtype = np.float32
        # import pdb
        # pdb.set_trace()
        if engine.binding_is_input(binding):
            input_buffer = np.ascontiguousarray(data)
            input_memory = cuda.mem_alloc(data.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

    stream = cuda.Stream()
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(input_memory, input_buffer, stream)
    # # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)

    # Synchronize the stream
    stream.synchronize()
    return output_buffer

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT')
    
    parser.add_argument('--device', help='cuda or not',
        default='cuda:0')
        
    # Sample image
    parser.add_argument('--batch_size', type=int, help='data batch size',
        default=16)
    parser.add_argument('--img_size', help='input size',
        default=[3, 224, 224])
    parser.add_argument('--sample_folder_path', help='sample image folder path',
        default='/home/asi/camera/thamnt/dataset/lp_color')
    parser.add_argument('--sample_image_path', help='sample image path',
        default='/home/asi/camera/thamnt/bank_security/out/wrong_mask/2/48777_04_25_46_823_497_mask_PUSH_PERIOD.jpg')

    # Model path
    parser.add_argument('--tensorrt_engine_path',  help='tensorrt engine path',
        default='/home/asi/camera/thamnt/bank_security/plugins/mask/mask_nomask_clf/mask_res34_final5_best.onnx_b16_gpu0_fp16.engine')

    # TensorRT engine params
    parser.add_argument('--dynamic_axes', help='dynamic batch input or output',
        default='True')
    parser.add_argument('--engine_precision', help='precision of TensorRT engine', choices=['FP32', 'FP16'], 
    	default='FP16')
    parser.add_argument('--min_engine_batch_size', type=int, help='set the min input data size of model for inference', 
    	default=1)
    parser.add_argument('--opt_engine_batch_size', type=int, help='set the most used input data size of model for inference', 
    	default=1)
    parser.add_argument('--max_engine_batch_size', type=int, help='set the max input data size of model for inference', 
    	default=8)
    parser.add_argument('--engine_workspace', type=int, help='workspace of engine', 
    	default=1024)
        
    args = string_to_bool(parser.parse_args())
    transforming = get_transform(args.img_size)
    classes = ['mask', 'nomask']
    # Read the engine from the file and deserialize
    with open(args.tensorrt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
        engine = runtime.deserialize_cuda_engine(f.read())    
    context = engine.create_execution_context()
    context.set_binding_shape(0, (args.batch_size, args.img_size[0], args.img_size[1], args.img_size[2]))
    # image inference
    img = cv2.imread(args.sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transforming(img).unsqueeze(0)
    print(img.shape)
    input = img.cpu().numpy().repeat(16, axis=0)
    print(input.shape)
    trt_outputs = trt_inference(engine, context, input)
    trt_outputs = np.array(trt_outputs)
    print(trt_outputs)
    pred = np.argmax(trt_outputs, axis=-1)
    print("Predict:", classes[int(pred)])

    # # batch inference
    # trt_start_time = time.time()
    # dataset = datasets.ImageFolder(args.sample_folder_path, transform=transforming)
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=args.batch_size,
    #                                          shuffle=True,
    #                                          num_workers=1)
    # # TensorRT inference
    # true_labels = []
    # pred_labels = []
    # for imgs, classes in tqdm(data_loader):
    #     imgs = imgs.cpu().numpy()
    #     trt_outputs = trt_inference(engine, context, imgs)
    #     trt_outputs = np.array(trt_outputs).reshape(args.batch_size, -1)
    #     preds = np.argmax(trt_outputs, axis=1)
    #     true_labels.extend(classes.tolist())
    #     pred_labels.extend(preds.tolist())
    # pred_labels = pred_labels[:len(true_labels)]
    # trt_end_time = time.time()
    # print("Accuracy: ", np.sum(np.array(true_labels)==np.array(pred_labels)) / len(true_labels))
    # print('Time: ', trt_end_time - trt_start_time)
    # from sklearn.metrics import confusion_matrix
    # classes = ['blue', 'red', 'white', 'yellow']
    # true_labels = [classes[int(i)] for i in true_labels]
    # pred_labels = [classes[int(i)] for i in pred_labels]
    # cmatrix = confusion_matrix(true_labels, pred_labels, labels = ['blue', 'red', 'white', 'yellow'])
    # print(['blue', 'red', 'white', 'yellow'])
    # print(cmatrix)
       
    
if __name__ == '__main__': 
    main()