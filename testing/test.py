import click
import os
from pathlib import Path
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png
import io, re
import numpy as np
import HouseConfig as Mapper

ROOT = Path('./cubicasa5k')
BRANCH = ['colorful', 'high_quality', 'high_quality_architectural']

IMAGE_FOLDER_PATH = Path('./testing/image')
LABEL_FOLDER_PATH = Path('./testing/label')

ROOM_COLOR = [
  (240, 180, 60),
  (180, 240, 120),
  (120, 60, 180),
  (60, 120, 240),
  (180, 60, 240), 
  (240, 120, 180), 
  (60, 180, 120), 
  (120, 240, 60), 
  (60, 240, 180),
  (120, 180, 240),
  (180, 120, 60),
  (240, 60, 120),
]

@click.command()
@click.option("--test-file",
              default='./testing/test.txt',
              type=str,
              help="test-file")
def main(test_file):
  with open(test_file) as f:
    dataname = f.read().split('\n')

    id = 1
    for name in dataname:
      file, svg = None, None
      for folder in BRANCH:
        path = ROOT / folder / name
        
        if not path.exists():
          continue

        file = path / 'F1_scaled.png'
        svg = path / 'model.svg'
        break
      
      if file is None or svg is None:
        continue

      img = Image.open(file)
      with open(svg) as f:
        svg = BeautifulSoup(f.read(), 'xml')

      modify_svg(svg)

      cvt = svg2pil(svg)
      label = Image.new("RGB", img.size)
      label.paste(cvt)

      filename = f'CUBI{id:02d}.png'

      img.save(IMAGE_FOLDER_PATH / filename)
      label.save(LABEL_FOLDER_PATH / filename)
      id += 1


def modify_svg(svg):
  for fp in svg.select("g[class=Floor]"):
    fp['style'] = ''

  for text_tag in svg.select('text'):
    text_tag.extract()

  for g in svg.select("g"):
    filt = "class" in g.attrs and "FixedFurniture" in g.attrs["class"] \
        or "id" in g.attrs    and "Panel" in g.attrs["id"] \
        or "id" in g.attrs    and "Stairs" in g.attrs["id"] \
        or "class" in g.attrs and "SelectionControls" in g.attrs["class"]

    if filt :
      g.extract()
    elif "class" in g.attrs and "Space " in g.attrs["class"]:
      alias_room_name = g.attrs['class'].split(' ')[1]
      name = Mapper.getRoomName(alias_room_name)
      ID = Mapper.getRoomID(name)

      g.attrs["fill"] = '#%02x%02x%02x' % ROOM_COLOR[ID]
      g.attrs["style"] = "fill-opacity: 1; stroke-opacity: 0; stroke-width: 0;"
      
  
  setStyle(svg, [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
  ])


def svg2pil(svg):
  mem = io.BytesIO()
  svg2png(bytestring=str(svg), write_to=mem)
  return Image.open(mem)


def setStyle(svg, color):
  style_directory = Path('./testing/boundaries.css')

  if not style_directory.exists():
    return
  
  s = ""
  with open(style_directory) as f:
    s = f.read()
  
  trans = {}
  for i,c in enumerate(color):
    trans[f'%%COLOR[{i}]%%'] = '#%02x%02x%02x' % c

  rep = dict((re.escape(k), v) for k, v in trans.items())
  pattern = re.compile("|".join(rep.keys()))
  s = pattern.sub(lambda m: rep[re.escape(m.group(0))], s)

  for style_tag in svg.select('style'):
    style_tag.extract()

  tag = svg.new_tag("style")
  tag.string = s
  svg.svg.g.insert_before(tag)


if __name__ == "__main__":
  main()