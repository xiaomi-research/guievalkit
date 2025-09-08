import ast
import json
import os

from pathlib import Path
from PIL import Image


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

root = Path(os.path.join(current_dir, "android_in_the_zoo/test"))
img_key = 'image_path'

for jp in root.rglob('*.json'):
    with open(jp, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    recs = obj if isinstance(obj, list) else [obj]

    for r in recs:
        if img_key not in r:
            continue

        img = root / Path(r[img_key])
        w, h = Image.open(img).size
        c = len(Image.open(img).getbands())
        r['image_height'] = h
        r['image_width'] = w
        r['image_channels'] = c

        if 'ui_positions' in r:
            try:
                pos = ast.literal_eval(r['ui_positions'])
                norm = [[y / h, x / w, hh / h, ww / w] for y, x, hh, ww in pos]
                r['ui_positions'] = json.dumps(norm, ensure_ascii=False)
            except Exception:
                pass

    with open(jp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

print('Done')
