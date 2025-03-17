import torch 
from ultralytics import YOLO
import os
import onnxruntime
import onnxruntime as ort
import cv2
import numpy as np
import pandas as pd
import time
from torchinfo import summary
import onnx, onnx_tool
import torchvision
from torch.utils.data import Dataset
import thop
from onnx_opcounter import calculate_params, calculate_macs
from PIL import Image
from codecarbon import OfflineEmissionsTracker

total_infer=0

def preprocessor(frame):
    x = cv2.resize(frame, (640, 640))
    image_data = np.array(x).astype(np.float32) / 255.0  # Normalize to [0, 1] >
    image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

    return image_data
class Inference:
    def __init__(self, model, path):
        model_bytes = model.SerializeToString()
        self.session = onnxruntime.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        self.path = path
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.classes = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5'}

    def detector(self, image_data):
        global total_infer 
        ort = onnxruntime.OrtValue.ortvalue_from_numpy(image_data)
        start_time = time.time()
#        for output in self.session.get_outputs():
#            print(f"Output name: {output.name}, shape: {output.shape}, type: {output.type}")
        results = self.session.run(["output0"], {"images": ort}) #991, input.1
        end_time = time.time()
        inference_time = end_time - start_time
        total_infer+=inference_time
        return results

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box
        color = (0, 255, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f'{self.classes[class_id]}: {score:.2f}'
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img


    def postprocessor(self, results, frame, confidence, iou):
        img_height, img_width = frame.shape[:2]
        outputs = np.transpose(np.squeeze(results[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        x_factor = img_width / self.input_width  # img_width = 640
        y_factor = img_height / self.input_height  # img_width = 640
        for i in range(rows):
            classes_scores = outputs[i][4:]

            max_score = np.amax(classes_scores)
            if max_score >= confidence:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
        sorted_boxes = [boxes[i] for i in sorted_indices]

        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_class_ids = [class_ids[i] for i in sorted_indices]

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(sorted_boxes, sorted_scores, confidence, iou)

        sorted_box=[]
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            frame = self.draw_detections(frame, box, score, class_id)
            sorted_box.append(box)
            
        return sorted_box

    def pipeline(self):
        frame = cv2.imread(self.path)
        frame = self.postprocessor(self.detector(preprocessor(frame)), frame, 0.3, 0.5) #dynamic 0.3 confidence
        return frame


def iou(box1, box2):
    """Compute the IoU of two bounding boxes."""
    # Coordinates of the intersection box
    xA = max(box1[0]/640, box2[0])
    yA = max(box1[1]/640, box2[1])
    xB = min(box1[2]/640, box2[2])
    yB = min(box1[3]/640, box2[3])
    
    
    # Area of overlap
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Area of both boxes
    box1Area = (box1[2]/640 - box1[0]/640 + 1) * (box1[3]/640 - box1[1]/640 + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Union Area
    unionArea = box1Area + box2Area - interArea
    
    # IoU
    return interArea / unionArea

def compute_ap50(pred_boxes, true_boxes, iou_threshold=0.5):
    """Compute AP@50 for a single class."""
    tp = 0
    fp = 0
    fn = 0


    # Sort predictions by confidence score (high to low)
#    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    matched_gt = []  # To keep track of ground truths that have been matched

    for pred_box in pred_boxes:
        best_iou = 0
        best_gt = None

        for gt_box in true_boxes:
#            print(matched_gt)
            if not any(np.array_equal(gt_box, matched) for matched in matched_gt):
#            if gt_box not in matched_gt:
                iou_value = iou(pred_box[:4], gt_box[:4])

                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt = gt_box
#            print(best_gt)

        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1
            matched_gt.append(best_gt)
        else:
            fp += 1
    
    fn = len(true_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall

tracker = OfflineEmissionsTracker(country_iso_code="CAN")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['WANDB_DISABLED'] = 'true'
path='/your/path/to/robotflow/test/images'
path_label='/your/path/to/robotflow/test/labels'
model = YOLO('/your/path/to/runs/detect/train2/weights/best.pt')
#model.export(format='onnx')

model_pruned ='/your/path/to/object_detection/original.onnx' 
model_pruned= onnx.load(model_pruned)
params = calculate_params(model_pruned)
#onnx_macs = calculate_macs(model_pruned) #doesn't work
print(f"Total number of parameters in the ONNX model: {params}")

x = torch.randn(1,3,640,640) #.cuda()
macs, params = thop.profile(model.model,inputs=(x,))
print('original model MACs:',macs,'params:',params)


precision1=0
recall1=0
tracker.start()
for index,filename in enumerate(sorted(os.listdir(path))):
    x=Inference(model_pruned,os.path.join(path,filename))
    y = x.pipeline()

    label=sorted(os.listdir(path_label))[index]
    try:
        content=pd.read_table(os.path.join(path_label,label),sep=' ',header=None,on_bad_lines='skip')    
    except pd.errors.EmptyDataError:
        content = pd.DataFrame()
    y_pred=y[:content.shape[0]]
    
    boxes=[]
    for i in range(content.shape[0]):

        bound_box=content.values[i,1:5]
        boxes.append(bound_box)
    
    
    precision, recall = compute_ap50(y_pred, boxes)

    precision1+=precision
#    recall1+=recall
precision_avg=precision1/(index+1)
#recall_avg=recall1/(index+1)
print('map50:',precision_avg)
print('infer time per image:',total_infer/(index+1))
tracker.stop()
