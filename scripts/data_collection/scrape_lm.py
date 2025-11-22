import requests
import constants
import json
from pathlib import Path
import time
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Scrape LearnableMeta data")
    parser.add_argument("--geoguessr-id", type=str, required=True,
                        help="GeoGuessr map ID")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for data")
    args = parser.parse_args()
    
    geoguessrId = args.geoguessr_id
    endpoint = f"https://learnablemeta.com/api/userscript/map/{geoguessrId}/"
    
    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True)
    folder = data_root / geoguessrId
    folder.mkdir(exist_ok=True)

## get metadata
md_file = folder / f"metadata_{geoguessrId}.json"
if not md_file.exists():
    headers = {"Content-Type": "application/json"}
    r = requests.get(endpoint, headers=headers)
    metadata = r.json()

    if metadata['mapFound'] != True:
        print(f"Map {geoguessrId} not found!")
        exit()

    with md_file.open('w') as f:
        json.dump(metadata, f)
else:
    with md_file.open() as f:
        metadata = json.load(f)

## get the actual locations of this map
loc_file = folder / f"locations_{geoguessrId}.json"
if not Path(loc_file).exists():
    headers = {"Authorization": f"Bearer {constants.LEARNABLE_META_KEY}",
            "Content-Type": "application/json"}

    r = requests.get(endpoint+"locations", headers=headers)
    location_data = r.json()

    with loc_file.open('w') as f:
        json.dump(location_data, f)
else:
    with loc_file.open() as f:
        location_data = json.load(f)

print(f"Got {len(location_data['customCoordinates'])} locations for map {geoguessrId}")

## get meta for all locations

meta_folder = folder / "metas"
meta_folder.mkdir(exist_ok=True)

seen_this_session = []
for loc in tqdm(location_data['customCoordinates']):
    info_endpoint = f"https://learnablemeta.com/api/userscript/location?"
    payload = {'panoId': loc['panoId'], 
            'mapId': geoguessrId, 
            "userscriptVersion": metadata['userscriptVersion'],
            "source": "map"
            }

    r = requests.get(info_endpoint, params=payload)
    meta = r.json()

    if not (meta_folder / loc['panoId']).exists():
        seen_this_session.append(loc['panoId'])
        with (meta_folder / (loc['panoId']+".json")).open('w') as f:
            json.dump(meta, f)
    else:
        if loc['panoId'] in seen_this_session: # only report if its during the current scraping session, otherwise probably just from prior run
            print(f"Collision for {loc['panoId']}, {str(meta)}")
        seen_this_session.append(loc['panoId'])

    time.sleep(0.5)

    print(f"Collected metas for {len(seen_this_session)} locations!")

if __name__ == "__main__":
    main()
