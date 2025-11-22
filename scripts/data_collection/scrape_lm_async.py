import aiohttp
import asyncio
import json
from pathlib import Path
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
import os

load_dotenv()
LEARNABLE_META_KEY = os.getenv("LEARNABLE_META_KEY")

def main():
    parser = argparse.ArgumentParser(description="Scrape LearnableMeta data (async)")
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

    # get metadata
    md_file = folder / f"metadata_{geoguessrId}.json"
    if not md_file.exists():
        import requests
        headers = {"Content-Type": "application/json"}
        r = requests.get(endpoint, headers=headers)
        r.raise_for_status()
        
        if not r.text or not r.text.strip():
            raise ValueError(f"Empty response from API endpoint: {endpoint}")
        
        try:
            metadata = r.json()
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response. Status: {r.status_code}")
            print(f"Response text (first 500 chars): {r.text[:500]}")
            raise
        
        if metadata.get('mapFound') != True:
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
        headers = {"Authorization": f"Bearer {LEARNABLE_META_KEY}",
                "Content-Type": "application/json"}
        r = requests.get(endpoint+"locations", headers=headers)
        r.raise_for_status()
        
        if not r.text or not r.text.strip():
            raise ValueError(f"Empty response from API endpoint: {endpoint}locations")
        
        try:
            location_data = r.json()
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response. Status: {r.status_code}")
            print(f"Response text (first 500 chars): {r.text[:500]}")
            raise
        
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
        try:
            async with session.get(info_endpoint, params=payload) as r:
                r.raise_for_status()
                text = await r.text()
                if not text or not text.strip():
                    print(f"Empty response for panoId: {loc['panoId']}")
                    return None
                try:
                    meta = json.loads(text)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON for panoId {loc['panoId']}: {e}")
                    print(f"Response (first 200 chars): {text[:200]}")
                    return None
                seen_this_session.append(loc['panoId'])
                with meta_path.open('w') as f:
                    json.dump(meta, f)
                return loc['panoId']
        except Exception as e:
            print(f"Error fetching meta for panoId {loc['panoId']}: {e}")
            return None

    async def fetch_all():
        connector = aiohttp.TCPConnector(limit=10)  # limit concurrency
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_meta(session, loc, metadata, meta_folder, geoguessrId, seen_this_session)
                     for loc in location_data['customCoordinates']]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await f

    asyncio.run(fetch_all())
    print(f"Collected metas for {len(seen_this_session)} locations!")

if __name__ == "__main__":
    main()
