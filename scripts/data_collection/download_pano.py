from streetview import get_panorama
from pathlib import Path
from tqdm import tqdm
from time import sleep

#geoguessrId = "6906237dc7731161a37282b2"
geoguessrId = "69061e1f2019c6dbc2c01a95"
meta_folder = Path(f"data/{geoguessrId}/metas/")

pano_folder = Path(f"data/{geoguessrId}/panorama/")
pano_folder.mkdir(exist_ok=True)

for fn in tqdm(meta_folder.glob("*.json")):
    if not (pano_folder / f"image_{fn.stem}.jpg").exists():
        #print(fn.stem)
        try:
            image = get_panorama(pano_id=fn.stem, zoom=4, multi_threaded=True)
            image.save(str(pano_folder / f"image_{fn.stem}.jpg"), "jpeg")
        except Exception as e:
            # unhandled exception (the streetview lib does automatic retries, so if it gets to here, its probably no good)
            print(e)
            print(f"Error for pano: {fn.stem}")
            
        sleep(0.1)
        #break
