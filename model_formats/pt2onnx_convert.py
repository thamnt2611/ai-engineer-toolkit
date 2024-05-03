import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# RESNET18
model_ft = models.resnet50(pretrained=False)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 4)
model_ft.load_state_dict(torch.load("/home/asi/camera/thamnt/clf_train/weapon_cls/models/clean_dup7_best.pt", map_location = device))

# # MOBILENET_V3
# model_ft = torchvision.models.mobilenet_v3_small()
# model_ft.classifier = nn.Sequential(*[model_ft.classifier[0], model_ft.classifier[1], model_ft.classifier[2], nn.Linear(1024, 4)])
# model_ft.load_state_dict(torch.load("/home/asi/camera/thamnt/plate_color_recognition/cnn/models/mobilenet/lp_color_model_mobilenetv3_best.pt", map_location = device))


model_ft.eval()
model_ft.to(device)
model = nn.Sequential(model_ft, nn.Softmax(dim = -1))
x = torch.randn(1, 3, 128, 128, requires_grad=False).to(device)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "/home/asi/camera/thamnt/clf_train/weapon_cls/models/clean_dup7_best.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

# from ultralytics import YOLO

# model = YOLO("/home/asi/camera/thamnt/vehicle_detection/models/yolov8sp_all_ep262_best.pt") 
# model.export(format="onnx", imgsz=[640,640], dynamic = True)
