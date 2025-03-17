from ultralytics import YOLO
from ultralytics import settings
import os
import torch.nn.utils.prune as prune
import torch
from torchinfo import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['WANDB_DISABLED'] = 'true'

ROOT_DIR = '/export/data/yliu/robotflow'
model = YOLO('/export/home/yliu/runs/detect/train2/weights/best.pt')
summary(model.model, input_size=(1, 3, 640, 640)) 
parameters_to_prune = []

for module in model.model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,  # This means pruning 20% of all parameters across the model
)


model.ckpt.update(dict(model=model.model))
del model.ckpt["ema"]
torch.save(model.model,"pruned_unstr_20.pt")
model.model=torch.load('pruned_unstr_20.pt')
results = model.train(data="/export/data/yliu/robotflow/data.yaml", epochs=200,batch=32,device=8,project='finetune',name='l1_pruning20')



# Evaluate the model's performance on the validation set
metrics = model.val()
model.export(format='onnx')



