import json
import base64

from PIL import Image
from io import BytesIO

import numpy as np
import re
import pickle

from .app.main import (
    roi_detection,
    wall_segmentation,
    symbol_detection
)
from .app.utils import (
    Image2Base64,
    Base642Image,
    ImageCropper,
    NumpyEncoder
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
        0: 'Room Classification'
    }
}


class PredictionConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code, raw_data=None):
        pass

    def receive(self, text_data):
        json_data = json.loads(text_data)
        ID = json_data["ID"]

        images = [
            Base642Image(json_data["image_data"])
        ]

        self.log('A0')
        roi_results = self.roiHandler(images)

        self.log('A1', {
            'name': 'roi-detection',
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

        with open('segment.pkl', 'wb') as f:
            pickle.dump(wall_segment_results, f)
        with open('symbol.pkl', 'wb') as f:
            pickle.dump(symbol_detection_results, f)

        post_processor = WallPostProcessing(self)
        for segment_results,symbol_results in zip(wall_segment_results, symbol_detection_results):
            for floor_segment,symbol_bbox in zip(segment_results, symbol_results):
                graphs, coords, coord_edges = post_processor(floor_segment, symbol_bbox)

        self.disconnect(0)


    def roiHandler(self, images: list[Image.Image]):
        return roi_detection(images)


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

    
    def format_image_data(self, name, image):
        if image.dtype.kind == "f" or image.dtype.kind == "b":
            image = (np.clip(image, a_min=0, a_max=1) * 255)
        image = image.astype(np.uint8)

        mode = "RGB"
        if len(image.shape) == 2 or image.shape[2] == 1:
            mode = "L"

        return {
            'name': name,
            'image': Image2Base64(
                Image.fromarray(image, mode)
            )
        }