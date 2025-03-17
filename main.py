from ultralytics import YOLO
from ultralytics import settings

import os
# Create a new YOLO model from scratch
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['WANDB_DISABLED'] = 'true'
#model = YOLO("yolov8n.yaml")
ROOT_DIR = '/your/path/to/robotflow'
# Load a pretrained YOLO model (recommended for training)
settings.update({"wandb":False})
settings.reset()

model = YOLO("yolov8n.pt")
#model = YOLO("yolov8n.yaml").load("yolov8n.pt")
#Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="/your/path/to/data.yaml", epochs=100,batch=32,device=6)

# Evaluate the model's performance on the validation set
metrics = model.val()

metrics = model.val(data="/your/path/to/data.yaml",split='test',device='6')
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category



