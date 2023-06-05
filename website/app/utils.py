from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import base64
import re, json
from pathlib import Path
import cv2


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


save_index = 0
def save(img):
    global save_index
    if img.dtype.kind == 'b':
        img = img.astype(np.int32)
    Image.fromarray((((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)).save(f'./saved/{save_index}.png')
    save_index += 1

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


model = ""
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
test_array = None
test_index = {}
test_path = Path('./testing/')
S = 512
def test(result, gnn):
  global test_array
  global test_index
  global test_path
  global S
  if test_array is None:
    init_test()
  
  if gnn not in test_index.keys():
    test_index[gnn] = 0
    with open(test_path / "result" / f"{gnn}.out", "w") as f:
      f.write('i/u\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\n')
  
  file = test_array[test_index[gnn]]
  test_index[gnn] += 1
  gt_img = np.array(Image.open(file))
  res_img = np.array(result)

  gt = np.zeros((gt_img.shape[0], gt_img.shape[1])) - 1
  res = np.zeros((res_img.shape[0], res_img.shape[1])) - 1
  
  for k in range(len(COLOR)):
    gt = np.where((gt_img[:,:,0] == COLOR[k][0]) & (gt_img[:,:,1] == COLOR[k][1]) & (gt_img[:,:,2] == COLOR[k][2]), k, gt)
    res = np.where((res_img[:,:,0] == COLOR[k][0]) & (res_img[:,:,1] == COLOR[k][1]) & (res_img[:,:,2] == COLOR[k][2]), k, res)
  
  gt = Image.fromarray(gt)
  res = Image.fromarray(res)

  new_s = [0, 0]
  if gt.size[0] > gt.size[1]:
    new_s[0] = int(S)
    new_s[1] = int(gt.size[1] / gt.size[0] * S)
  else:
    new_s[1] = int(S)
    new_s[0] = int(gt.size[0] / gt.size[1] * S)
  
  gt = np.array(gt.resize(tuple(new_s), Image.NEAREST))
  res = np.array(res.resize(tuple(new_s), Image.NEAREST))

  save(gt)
  save(res)

  intersection = np.zeros(len(COLOR), dtype=np.int32)
  union = np.zeros(len(COLOR), dtype=np.int32)
  
  for i in range(len(COLOR)):
    intersection[i] += np.sum(np.logical_and((gt == i), (res == i)))
    union[i] += np.sum(np.logical_or((gt == i), (res == i)))
  
  with open(test_path / "result" / f"{gnn}.out", "a") as f:
    for i in intersection:
      f.write(f"\t{i}")
    for i in union:
      f.write(f"\t{i}")
    f.write("\n")
  print("done")


def init_test():
  global test_array
  test_array = []
  root = Path('./testing')
  for i in range(40):
    k = i + 1
    test_array.append(root / 'label' / f'CUBI{k:02d}.png')
  for i in range(10):
    k = i + 1
    test_array.append(root / 'label' / f'TEXT{k:02d}.png')