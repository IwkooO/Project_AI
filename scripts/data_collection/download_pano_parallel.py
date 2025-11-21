from streetview import get_panorama
from pathlib import Path
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

geoguessrId = "691df1ee911f74393c53af8c"
meta_folder = Path(f"data/{geoguessrId}/metas/")
pano_folder = Path(f"data/{geoguessrId}/panorama/")
pano_folder.mkdir(exist_ok=True)

def download_pano(fn):
    pano_path = pano_folder / f"image_{fn.stem}.jpg"
    if pano_path.exists():
        return None
    try:
        image = get_panorama(pano_id=fn.stem, zoom=4, multi_threaded=True)
        image.save(str(pano_path), "jpeg")
    except Exception as e:
        print(e)
        print(f"Error for pano: {fn.stem}")
    sleep(0.1)
    return fn.stem

files = list(meta_folder.glob("*.json"))
with ThreadPoolExecutor(max_workers=64) as executor:  # Adjust max_workers as needed
    futures = [executor.submit(download_pano, fn) for fn in files]
    for _ in tqdm(as_completed(futures), total=len(futures), 
                  desc="Downloading", disable=False, file=sys.stdout, 
                  miniters=1, mininterval=1.0):
        pass
