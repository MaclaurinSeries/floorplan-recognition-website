import json
import base64

from PIL import Image, ImageDraw
from io import BytesIO
from pathlib import Path

import numpy as np
import re, os
import pickle

from torch_geometric.data import Data
from .app import HouseConfig as Mapper
from copy import deepcopy

from shapely.geometry import Polygon

from .app.main import (
    roi_detection,
    wall_segmentation,
    symbol_detection,
    room_clasification,
    spatial_classification,
    text_detection
)
from .app.utils import (
    Image2Base64,
    Base642Image,
    ImageCropper,
    NumpyEncoder,
    test
)
from .app.segmentationPostProcess import (
    WallPostProcessing
)

from channels.generic.websocket import WebsocketConsumer

PROCESS_ID = {
    'A': {
        0: 'ROI Detection',
        1: 'Floor Splitting'
    },
    'B': {
        0: 'Symbol Detection',
        1: 'Text Detection'
    },
    'C': {
        0: 'Wall Segmentation',
        1: '(post..) skeletoning',
        2: '(post..) vectorization',
        3: '(post..) graph construction'
    },
    'D': {
        0: 'Deep Floor Plan',
        1: 'Text Detection',
        2: 'Room Classification'
    },
    'E': {
        0: 'Floor Result'
    }
}

COLOR = [
  (0, 255, 0),
  (255, 0, 0),
  (0, 0, 255),
  (240, 180, 60),
  (180, 240, 120),
  (120, 60, 180),
  (60, 120, 240),
  (180, 60, 240), 
  (240, 120, 180), 
  (60, 180, 120), 
  (120, 240, 60), 
  (60, 240, 180),
  (120, 180, 240)
]


class PredictionConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code, raw_data=None):
        pass

    def receive(self, text_data):
      json_data = json.loads(text_data)
      ID = json_data["ID"]
      gnn = json_data["gnn"]

      images = [
        Base642Image(json_data["image_data"])
      ]

      #testing
      # test_array = []
      # root = Path('./testing')
      # for i in range(40):
      #   k = i + 1
      #   test_array.append(root / 'image' / f'CUBI{k:02d}.png')
      # for i in range(10):
      #   k = i + 1
      #   test_array.append(root / 'image' / f'TEXT{k:02d}.png')
      # for f in test_array:
      #   images = [
      #     Image.open(f)
      #   ]
      #   first_shape = images[0].size

      self.log('A0')
      roi_results = self.roiHandler(images)

      self.log('A1', {
        'name': 'roi-detection',
        'floor': len(roi_results[0]),
        'preds': roi_results[0]
      })

      #divide each floor
      images_divided = []
      for image,rois in zip(images,roi_results):
        images_divided.append(
          [
            ImageCropper(image, roi) for roi in rois
          ]
        )

      self.log('B0')
      symbol_detection_results = self.symbolDetectionHandler(images_divided)

      self.log('C0', {
        'name': 'symbol-detection',
        'preds': symbol_detection_results[0]
      })
      wall_segment_results = self.wallSegmentationHandler(images_divided)

      # with open('segment.pkl', 'wb') as f:
      #     pickle.dump(wall_segment_results, f)
      # with open('symbol.pkl', 'wb') as f:
      #     pickle.dump(symbol_detection_results, f)

      post_processor = WallPostProcessing(self)
      # all_poly = []
      # all_pred = []
      # all_coords = []
      # all_ppm = []
      for segment_results,symbol_results,images in zip(wall_segment_results, symbol_detection_results, images_divided):
        # al_poly = []
        # al_pred = []
        # al_coords = []
        # al_ppm = []
        for floor_idx,(floor_segment,symbol_bbox,image) in enumerate(zip(segment_results, symbol_results, images)):
          floor_idx += 1
          graph, room_poly, coords, coord_edges, ppm = post_processor(floor_segment, symbol_bbox, floor_idx)
          location = []
          # al_poly.append(room_poly)
          # al_coords.append((coords, coord_edges))
          # al_ppm.append(ppm)
          for pol in room_poly:
            pts = deepcopy(pol)
            pts[0,:] *= image.size[1]
            pts[1,:] *= image.size[0]
            p = Polygon(pts.T).centroid
            location.append([p.x / image.size[1], p.y / image.size[0]])

          self.log('D0', {
            'name': 'graph-construction',
            'floor': floor_idx,
            'room_poly': room_poly,
            'x_location': location,
            'edge': graph.edge_index.cpu().numpy(),
            'door_edges': coord_edges
          })
          x_cnn,res = spatial_classification(image, room_poly)
          self.log('D1', self.format_image_data(
            name='deep-floor-plan',
            floor=floor_idx,
            image=res
          ))
          # text detection
          x_text, text_bounding_box = text_detection(image, room_poly)
          self.log('D2', {
            'name': 'text-detection',
            'floor': floor_idx,
            'preds': text_bounding_box
          })
          preds = self.gnnHandler(graph, gnn, x_cnn, x_text)
          self.log('E0', {
            'name': 'room-classification',
            'floor': floor_idx,
            'preds': preds.argmax(-1),
            'labels': Mapper.rooms
          })
          #   al_pred.append(preds.argmax(-1))
          # all_poly.append(al_poly)
          # all_pred.append(al_pred)
          # all_coords.append(al_coords)
          # all_ppm.append(al_ppm)
        
        # test_img = Image.new("RGB", first_shape)
        # drawer = ImageDraw.Draw(test_img)
        # for roi,poly,pred in zip(roi_results[0], all_poly[0], all_pred[0]):
        #   bbox = roi['bounding_box']
        #   for po,pr in zip(poly,pred):
        #     color = COLOR[pr + 3]
        #     po[0] = bbox[1] + po[0] * (bbox[3] - bbox[1])
        #     po[1] = bbox[0] + po[1] * (bbox[2] - bbox[0])

        #     plo = []
        #     for py in po[::-1,:].astype(int).T:
        #       plo.append((py[0], py[1]))

        #     drawer.polygon(plo, outline=0, fill=color)
        # for roi,(coords,coord_edges),ppm in zip(roi_results[0], all_coords[0], all_ppm[0]):
        #   bbox = roi['bounding_box']
        #   for edge in coord_edges:
        #     for pl in edge['polygon']:
        #       poly = []
        #       for ar in pl:
        #         poly.append((int(bbox[0] + ar[0] * (bbox[2] - bbox[0])), int(bbox[1] + ar[1] * (bbox[3] - bbox[1]))))
        #       if len(poly) > 1:
        #         drawer.polygon(poly, outline=0, fill=COLOR[0])

        #     for door in edge['doors']:
        #       poly = []
        #       for ar in door['polygon']:
        #         poly.append((int(bbox[0] + ar[0] * (bbox[2] - bbox[0])), int(bbox[1] + ar[1] * (bbox[3] - bbox[1]))))
        #       if len(poly) > 1:
        #         drawer.polygon(poly, outline=0, fill=COLOR[1])
        #     for window in edge['windows']:
        #       poly = []
        #       for ar in window['polygon']:
        #         poly.append((int(bbox[0] + ar[0] * (bbox[2] - bbox[0])), int(bbox[1] + ar[1] * (bbox[3] - bbox[1]))))
        #       if len(poly) > 1:
        #         drawer.polygon(poly, outline=0, fill=COLOR[2])
        # test_img.save('test.png')
        # test(test_img, gnn)

        self.disconnect(0)


    def roiHandler(self, images: list[Image.Image]):
        return roi_detection(images)
    
    
    def gnnHandler(self, graph: Data, model_name, x_cnn, x_text):
        x_compose = x_cnn + x_text
        return room_clasification(graph, x_compose, model_name)


    def wallSegmentationHandler(self, images: list[list[Image.Image]]):
        image_CNT = [len(sublist) for sublist in images]
        image_CNT_pref_sum = [sum(image_CNT[:i]) for i in range(len(image_CNT))]

        crops = [item for sublist in images for item in sublist]
        seg_results = wall_segmentation(crops)
        results = [
            seg_results[pref_sum: pref_sum + image_CNT[i]] for i,pref_sum in enumerate(image_CNT_pref_sum)
        ]

        return results
    

    def symbolDetectionHandler(self, images: list[list[Image.Image]]):
        image_CNT = [len(sublist) for sublist in images]
        image_CNT_pref_sum = [sum(image_CNT[:i]) for i in range(len(image_CNT))]

        crops = [item for sublist in images for item in sublist]
        sd_results = symbol_detection(crops)
        results = [
            sd_results[pref_sum: pref_sum + image_CNT[i]] for i,pref_sum in enumerate(image_CNT_pref_sum)
        ]

        return results


    def log(self, pid: str, data=None):
        alpha = pid[0]
        beta = int(pid[1])
        info = PROCESS_ID[alpha][beta]

        alpha_process = list(PROCESS_ID)
        alpha_part = 100 / len(alpha_process)
        beta_process = list(PROCESS_ID[alpha])
        beta_part = alpha_part / len(beta_process)

        process_percentage = alpha_part * alpha_process.index(alpha) + beta_part * beta_process.index(beta)

        self.send(text_data=json.dumps({
            'PID': pid,
            'percentage': process_percentage,
            'info': info,
            'data': data
        }, cls=NumpyEncoder))

    
    def format_image_data(self, name, floor, image):
        if image.dtype.kind == "f" or image.dtype.kind == "b":
            image = (np.clip(image, a_min=0, a_max=1) * 255)
        image = image.astype(np.uint8)

        mode = "RGB"
        if len(image.shape) == 2 or image.shape[2] == 1:
            mode = "L"

        return {
            'name': name,
            'floor': floor,
            'image': Image2Base64(
                Image.fromarray(image, mode)
            )
        }