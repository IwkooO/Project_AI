from streetview import get_panorama
from pathlib import Path
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download panoramas in parallel")
    parser.add_argument("--geoguessr-id", type=str, default="6906237dc7731161a37282b2",
                        help="GeoGuessr map ID")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for data")
    args = parser.parse_args()
    
    geoguessrId = args.geoguessr_id
    data_root = Path(args.data_root)
    meta_folder = data_root / geoguessrId / "metas"
    pano_folder = data_root / geoguessrId / "panorama"
    pano_folder.mkdir(parents=True, exist_ok=True)

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
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
        futures = [executor.submit(download_pano, fn) for fn in files]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == "__main__":
    main()

