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
from argparse import Namespace
from torch_geometric.data import Data
from .models.GAT import Model as GAT
from .models.GCN import Model as GCN
from .models.GraphSAGE import Model as GraphSAGE
from .models.GraphSAGEjk import Model as GraphSAGEjk
from .models.GCNjk import Model as GCNjk
from .dfp.deploy import main as deepfloorplan
import pytesseract
import cv2

roi_detection_model_path = Path() / 'model-collection' / 'roi-detection' / 'best.pt'
wall_segmentation_model_path = Path() / 'model-collection' / 'wall-segmentation' / 'weights.pt'
symbol_detection_model_path = Path() / 'model-collection' / 'symbol-detection' / 'best.pt'
room_classification_model_path = Path() / 'model-collection' / 'room-classification'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

NODE_FEATURE_LENGTH = 19
NODE_CLASS_LENGTH = 10
BATCH_SIZE = 128

dfp_floorplan_fusemap = [[0, 0, 0], [192, 192, 224], [192, 255, 255], [224, 255, 192], [255, 224, 128], [255, 160, 96], [255, 224, 224], [224, 224, 224], [224, 224, 128], [255, 60, 128], [255, 255, 255]]
dfp_floorplan_label = [[-1], [6], [4], [1, 2, 9], [3], [5], [8], [-1], [-1], [-1], [-1]]

dictionary_room_classification = {
    'gat': {
        'model': GAT,
        'best': 0,
    },
    'gcn': {
        'model': GCN,
        'best': 0,
    },
    'graph-sage': {
        'model': GraphSAGE,
        'best': 0,
    },
    'graph-sage-jk': {
        'model': GraphSAGEjk,
        'best': 0,
    },
    'gcn-jk': {
        'model': GCNjk,
        'best': 0,
    },
}


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

    preds = []
    with torch.no_grad():
        for image in images:
            pred = model(transformation(image).to(device)[None,:,:,:])
            preds.append(pred['out'][0,0])

    masks = []
    for i in range(input_LEN):
        mask = np.clip(preds[i].cpu().numpy(), a_min=0, a_max=1)
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


def room_clasification(graph: Data, x_compose, model_name, model_no=None):
    assert model_name in dictionary_room_classification.keys()

    graph.x[:,9:] = torch.from_numpy(x_compose)
    print(graph.x)

    if model_no is None:
        model_no = dictionary_room_classification[model_name]['best']
    model_path = room_classification_model_path / model_name / str(model_no) / 'weight.ckpt'

    model = dictionary_room_classification[model_name]['model'].load_from_checkpoint(
        model_path,
        map_location=device,
        in_channels=NODE_FEATURE_LENGTH,
        out_channels=NODE_CLASS_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    model.eval()
    with torch.no_grad():
        y_hat = model(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr
        )
        pred = y_hat.softmax(dim=-1)

    return pred.cpu().numpy()


def spatial_classification(image: Image.Image, room_poly, postprocess=False):
    args = Namespace(
        weight='./model-collection/deep-floor-plan/log/store/G',
        loadmethod='log',
        postprocess=postprocess,
        colorize=True,
        save=None
    )
    w,h = image.size
    res = deepfloorplan(args, image)
    N = len(room_poly)
    x = np.zeros((N, NODE_CLASS_LENGTH), dtype=np.float32)
    
    idf = np.zeros(NODE_CLASS_LENGTH, dtype=np.int32)

    for i,poly in enumerate(room_poly):
        mask = np.zeros((h, w, 1), dtype=np.int32)
        poly[0,:] *= h
        poly[1,:] *= w
        points = np.array([poly[::-1,:].T], dtype=np.int32)
        
        cv2.fillPoly(mask, [points], (255))
        arg = np.where((mask[:,:,0] > 0))

        colors, cnt = np.unique(res[arg], axis=0, return_counts=True)
        all_pixs = cnt.sum()

        for color,ct in zip(colors, cnt):
            pred = dfp_floorplan_fusemap.index(color.tolist())
            value = len(dfp_floorplan_fusemap[pred])
            for lbl in dfp_floorplan_label[pred]:
                idf[lbl] += ct / all_pixs
                if lbl >= 0:
                    x[i,lbl] = ct / (all_pixs * value)
    
    idf = np.log(N / (idf + 1))
    x = np.apply_along_axis(lambda r: r * idf, 1, x)
    
    return x, res


def text_detection(image: Image.Image):
    extracted = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
