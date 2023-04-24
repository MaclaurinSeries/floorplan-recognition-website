from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import base64
import re, json


def Image2Base64(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format='png')
    buffer.seek(0)

    data_uri = base64.b64encode(buffer.read()).decode('ascii')

    return f'data:image/png;base64,{data_uri}'

def Base642Image(uri: str):
    image_data = re.sub('^data:image/.+;base64,', '', uri)
    return Image.open(BytesIO(base64.b64decode(image_data)))

def ImageCropper(image: Image.Image, data):
    img = image.copy().convert("RGBA")
    polygon_mask = Image.new('RGBA', img.size, '#ffffffff')
    polygon_mask_draw = ImageDraw.Draw(polygon_mask)

    bbox = data['bounding_box']
    polygon = data['polygon']
    polygon = Polygon(polygon).buffer(0.1, cap_style=3) # round 1, flat 2, square 3

    if isinstance(polygon, Polygon):
        xy = np.array(polygon.exterior.xy).T
        polygon_mask_draw.polygon([(p[0], p[1]) for p in xy], fill='#00000000', width=1)
    elif isinstance(polygon, MultiPolygon):
        for p in polygon.geoms:
            if isinstance(p, Polygon):
                xy = np.array(p.exterior.xy).T
                polygon_mask_draw.polygon([(p[0], p[1]) for p in xy], fill='#00000000', width=1)

    return Image.alpha_composite(img, polygon_mask).convert("RGB").crop(bbox)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)