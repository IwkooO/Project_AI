import aiohttp
import asyncio
import json
from pathlib import Path
import sys
# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import constants
from tqdm import tqdm

geoguessrId = "6921d7831744c5356b098bf7"
endpoint = f"https://learnablemeta.com/api/userscript/map/{geoguessrId}/"

data_root = Path("data")
data_root.mkdir(exist_ok=True)
folder = data_root / geoguessrId
folder.mkdir(exist_ok=True)

# get metadata
md_file = folder / f"metadata_{geoguessrId}.json"
if not md_file.exists():
    import requests
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

# get the actual locations of this map
loc_file = folder / f"locations_{geoguessrId}.json"
if not Path(loc_file).exists():
    import requests
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

meta_folder = folder / "metas"
meta_folder.mkdir(exist_ok=True)

seen_this_session = []

async def fetch_meta(session, loc, metadata, meta_folder, geoguessrId, seen_this_session):
    info_endpoint = f"https://learnablemeta.com/api/userscript/location?"
    payload = {
        'panoId': loc['panoId'],
        'mapId': geoguessrId,
        "userscriptVersion": metadata['userscriptVersion'],
        "source": "map"
    }
    meta_path = meta_folder / (loc['panoId'] + ".json")
    if meta_path.exists():
        if loc['panoId'] in seen_this_session:
            print(f"Collision for {loc['panoId']}")
        seen_this_session.append(loc['panoId'])
        return None
    async with session.get(info_endpoint, params=payload) as r:
        meta = await r.json()
        seen_this_session.append(loc['panoId'])
        with meta_path.open('w') as f:
            json.dump(meta, f)
        return loc['panoId']

async def main():
    connector = aiohttp.TCPConnector(limit=10)  # limit concurrency
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_meta(session, loc, metadata, meta_folder, geoguessrId, seen_this_session)
                 for loc in location_data['customCoordinates']]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f

if __name__ == "__main__":
    asyncio.run(main())
    print(f"Collected metas for {len(seen_this_session)} locations!")
