from PIL import Image, ImageDraw
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path
from skimage.morphology import medial_axis
from .segmentationPostProcess import WallPostProcessing
import numpy as np
import torch

roi_detection_model_path = Path() / 'model-collection' / 'roi-detection' / 'best.pt'
wall_segmentation_model_path = Path() / 'model-collection' / 'wall-segmentation' / 'weights.pt'
symbol_detection_model_path = Path() / 'model-collection' / 'symbol-detection' / 'best.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def roi_detection(images: list[Image.Image]):
    model = YOLO(roi_detection_model_path)
    model.to(device)
    results = model(images)

    prediction_results = []

    for result in results:
        result = result.cpu()
        prediction_result = []

        bounding_box = result.boxes.xyxy.numpy() # numpy array, dtype float32
        polygon = result.masks.xy # numpy array, dtype float32
        polygon_n = result.masks.xyn

        for poly, bbox, polyn in zip(polygon, bounding_box, polygon_n):
            prediction_result.append({
                'polygon': poly,
                'polygon_n': polyn,
                'bounding_box': bbox
            })
        prediction_results.append(prediction_result)

    return prediction_results


def wall_segmentation(images: list[Image.Image]):
    input_LEN = len(images)

    model = models.segmentation.deeplabv3_resnet50()
    model.classifier = DeepLabHead(2048, 1)
    model.load_state_dict(torch.load(wall_segmentation_model_path, map_location=device).state_dict())
    model.eval()

    model.to(device)

    transformation = transforms.Compose([transforms.ToTensor()])

    input_images = torch.stack([transformation(image).to(device) for image in images])

    with torch.no_grad():
        pred = model(input_images)
    
    masks = []
    for i in range(input_LEN):
        mask = np.clip(pred['out'][i,0].cpu().numpy(), a_min=0, a_max=1)
        masks.append(mask)

    return masks


def symbol_detection(images: list[Image.Image]):
    model = YOLO(symbol_detection_model_path)
    model.to(device)
    results = model(images)

    prediction_results = []

    for result in results:
        result = result.cpu()
        prediction_result = []

        lbl_names = result.names
        cls_label = result.boxes.cls.numpy().astype(np.int32).tolist()
        cls_prob = result.boxes.conf.numpy().astype(np.float32).tolist()
        bounding_box = result.boxes.xyxy.numpy()
        bounding_box_n = result.boxes.xyxyn.numpy()

        for bbox,bboxn,label,prob in zip(bounding_box, bounding_box_n, cls_label, cls_prob):
            prediction_result.append({
                'bounding_box': bbox,
                'bounding_box_n': bboxn,
                'label': label,
                'prob': prob,
                'names': lbl_names[label]
            })
        prediction_results.append(prediction_result)

    return prediction_results