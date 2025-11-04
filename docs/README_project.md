# Outline

This code scrapes information about Geoguessr Metas from learnablemeta.com. This only works for maps that you have created yourself, however, since it is easy to copy metas from other maps it is trivial to do so.

Steps:
- Obtain your learnable meta API key and add it into constants.py
- Create a personal map on learnablemeta.com (this requires making a map on geoguessr, as a geoguessrid is needed)
- Go to the learnablemeta maps you'd like to import the metas from, and add their metas to your map
- (This step might be optional, didn't test if it works without) Go to your geoguessr map and import (using the learnablemeta userscript) the locations into your geoguessr map
- Ensure scripts/data_collection/scrape_lm.py contains the right geoguessrId for your map, and then run it.
- To download the panorama images associated with each meta: 
    - Ensure scripts/data_collection/download_pano.py has correct geoguessrId for your map
    - run download_pano.py

# Installation

I ran `conda env export > environment.yml` and `pip freeze > requirements.txt`, so you should be able to install from those, with `conda env create -f environment.yml` and `pip install -r requirements.txt`.
