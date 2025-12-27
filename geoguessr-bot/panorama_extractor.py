"""
Panorama ID Extractor for GeoGuessr.

Uses Selenium to extract the current panorama ID from GeoGuessr,
then downloads the actual Street View panorama for better predictions.
"""

import os
import io
import math
import requests
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

# Try to import selenium, but don't fail if not available
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not installed. Install with: pip install selenium")


class PanoramaExtractor:
    """Extract panorama ID from GeoGuessr and download Street View images."""
    
    def __init__(self, headless: bool = True):
        """
        Initialize the panorama extractor.
        
        Args:
            headless: If True, run Chrome in headless mode (no visible window)
        """
        self.driver = None
        self.headless = headless
        
    def connect_to_existing_chrome(self, debug_port: int = 9222) -> bool:
        """
        Connect to an existing Chrome instance with remote debugging enabled.
        
        To enable remote debugging, start Chrome with:
            google-chrome --remote-debugging-port=9222
        
        Args:
            debug_port: The debugging port Chrome is listening on
            
        Returns:
            True if connected successfully
        """
        if not SELENIUM_AVAILABLE:
            print("❌ Selenium not available")
            return False
            
        options = Options()
        options.add_experimental_option("debuggerAddress", f"127.0.0.1:{debug_port}")
        
        try:
            self.driver = webdriver.Chrome(options=options)
            print(f"✅ Connected to Chrome on port {debug_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Chrome: {e}")
            print("   Start Chrome with: google-chrome --remote-debugging-port=9222")
            return False
    
    def get_panorama_id(self) -> Optional[str]:
        """
        Extract the current panorama ID from the GeoGuessr page.

        Returns:
            The panorama ID string, or None if not found
        """
        if self.driver is None:
            print("❌ Not connected to browser")
            return None

        # Check if driver is still connected
        try:
            self.driver.current_url  # Simple check
        except Exception:
            print("❌ Browser connection lost")
            return None
        
        # Try multiple methods to extract pano ID
        pano_id = None
        
        # Method 1: Extract from Google Maps URL in iframe
        try:
            # GeoGuessr embeds Street View in an iframe
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                src = iframe.get_attribute("src") or ""
                if "maps.google.com" in src or "streetviewpixels" in src:
                    # Extract pano ID from URL parameters
                    if "pano=" in src:
                        pano_id = src.split("pano=")[1].split("&")[0]
                        break
        except Exception as e:
            print(f"Method 1 failed: {e}")
            return None
        
        # Method 2: Execute JavaScript to get pano ID from Google Maps API
        if pano_id is None:
            try:
                # Try to access the Street View panorama object
                js_code = """
                // Try to find panorama ID from various sources

                // Method A: From google.maps.StreetViewPanorama
                if (typeof google !== 'undefined' && google.maps) {
                    var svContainers = document.querySelectorAll('[class*="panorama"]');
                    for (var i = 0; i < svContainers.length; i++) {
                        var container = svContainers[i];
                        // Check if this element has a __gm property (Google Maps internal)
                        if (container.__gm && container.__gm.pano) {
                            return container.__gm.pano.getPano();
                        }
                    }
                }

                // Method B: From network requests (look for cbk URLs)
                var performance = window.performance || {};
                var entries = performance.getEntriesByType ? performance.getEntriesByType('resource') : [];
                for (var i = entries.length - 1; i >= 0; i--) {
                    var url = entries[i].name;
                    if (url.includes('cbk') && url.includes('panoid=')) {
                        var match = url.match(/panoid=([^&]+)/);
                        if (match) return match[1];
                    }
                }

                // Method C: From page source
                var scripts = document.querySelectorAll('script');
                for (var i = 0; i < scripts.length; i++) {
                    var text = scripts[i].textContent || '';
                    var match = text.match(/"panoId":"([^"]+)"/);
                    if (match) return match[1];
                }

                return null;
                """
                pano_id = self.driver.execute_script(js_code)
            except Exception as e:
                print(f"Method 2 failed: {e}")
                return None
        
        # Method 3: Look in page source for pano ID pattern
        if pano_id is None:
            try:
                page_source = self.driver.page_source
                import re

                # Look for pano ID patterns
                patterns = [
                    r'"panoId":"([A-Za-z0-9_-]+)"',
                    r'panoid=([A-Za-z0-9_-]+)',
                    r'"pano":"([A-Za-z0-9_-]+)"',
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, page_source)
                    if matches:
                        # Return the most recent/last match (likely current pano)
                        pano_id = matches[-1]
                        break
            except Exception as e:
                print(f"Method 3 failed: {e}")
                return None
        
        if pano_id:
            print(f"📍 Found panorama ID: {pano_id}")
        else:
            print("⚠️ Could not extract panorama ID")
            
        return pano_id
    
    def close(self):
        """Close the Selenium driver."""
        if self.driver:
            # Don't close if we connected to existing Chrome
            # self.driver.quit()
            self.driver = None


def download_streetview_panorama(
    pano_id: str,
    output_path: Optional[Path] = None,
    width: int = 640,
    height: int = 640,
    heading: float = 0,
    pitch: float = 0,
    fov: int = 90,
) -> Optional[Image.Image]:
    """
    Download a Street View image using the panorama ID.
    
    Note: This uses the unofficial Street View tile API which doesn't require an API key.
    For production use, consider using the official Google Street View Static API.
    
    Args:
        pano_id: The panorama ID
        output_path: Optional path to save the image
        width: Image width
        height: Image height
        heading: Camera heading (0-360)
        pitch: Camera pitch (-90 to 90)
        fov: Field of view (1-120)
        
    Returns:
        PIL Image if successful, None otherwise
    """
    # Method 1: Use Street View tiles directly (no API key needed)
    try:
        # Street View tile URL pattern
        # This fetches a single tile from the panorama
        zoom = 2  # 0-5, higher = more detail but more tiles needed
        
        # For a simple single image, we can use the cbk endpoint
        url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x=0&y=0"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            print(f"✅ Downloaded panorama tile: {img.size}")
            
            if output_path:
                img.save(output_path)
                print(f"   Saved to: {output_path}")
            
            return img
    except Exception as e:
        print(f"Tile download failed: {e}")
    
    # Method 2: Stitch multiple tiles for full panorama
    try:
        img = download_full_panorama(pano_id, zoom=2)
        if img is not None:
            # Resize to target dimensions
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            if output_path:
                img.save(output_path)
                print(f"   Saved to: {output_path}")
            
            return img
    except Exception as e:
        print(f"Full panorama download failed: {e}")
    
    return None


def download_full_panorama(pano_id: str, zoom: int = 2) -> Optional[Image.Image]:
    """
    Download and stitch a full 360° panorama from tiles.
    
    Args:
        pano_id: The panorama ID
        zoom: Zoom level (0-5). Higher = more detail but slower.
              zoom=2 gives 4x2 tiles = reasonable quality
              
    Returns:
        Stitched panorama image
    """
    # Tile dimensions at different zoom levels
    # zoom 0: 1x1 tiles, 512x512 each
    # zoom 1: 2x1 tiles
    # zoom 2: 4x2 tiles
    # zoom 3: 8x4 tiles
    # zoom 4: 16x8 tiles
    
    tiles_x = 2 ** zoom
    tiles_y = 2 ** (zoom - 1) if zoom > 0 else 1
    tile_size = 512
    
    # Create output image
    pano_width = tiles_x * tile_size
    pano_height = tiles_y * tile_size
    panorama = Image.new('RGB', (pano_width, pano_height))
    
    print(f"📷 Downloading {tiles_x}x{tiles_y} tiles for panorama...")
    
    success_count = 0
    for y in range(tiles_y):
        for x in range(tiles_x):
            url = f"https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    panorama.paste(tile, (x * tile_size, y * tile_size))
                    success_count += 1
            except Exception as e:
                print(f"   Failed to download tile ({x},{y}): {e}")
    
    if success_count == 0:
        print("❌ No tiles downloaded")
        return None
    
    print(f"✅ Downloaded {success_count}/{tiles_x * tiles_y} tiles")
    return panorama


def extract_view_from_panorama(
    panorama: Image.Image,
    heading: float = 0,
    pitch: float = 0,
    fov: float = 90,
    output_size: Tuple[int, int] = (640, 640),
) -> Image.Image:
    """
    Extract a view from a 360° panorama at a specific heading/pitch.
    
    This simulates what the user sees in Street View.
    
    Args:
        panorama: Full 360° panorama image
        heading: Horizontal angle (0-360, 0=North)
        pitch: Vertical angle (-90 to 90)
        fov: Field of view in degrees
        output_size: Output image size (width, height)
        
    Returns:
        Extracted view image
    """
    pano_width, pano_height = panorama.size
    out_width, out_height = output_size
    
    # Convert heading to x coordinate in panorama
    # heading 0 = center, 90 = 1/4 right, etc.
    x_center = (heading / 360.0) * pano_width
    
    # Convert FOV to crop width
    crop_width = int((fov / 360.0) * pano_width)
    crop_height = int(crop_width * (out_height / out_width))
    
    # Calculate crop region
    x1 = int(x_center - crop_width / 2)
    x2 = int(x_center + crop_width / 2)
    
    # Handle pitch (simplified - just vertical offset)
    y_center = pano_height / 2 - (pitch / 90.0) * (pano_height / 2)
    y1 = int(y_center - crop_height / 2)
    y2 = int(y_center + crop_height / 2)
    
    # Clamp y coordinates
    y1 = max(0, min(pano_height - crop_height, y1))
    y2 = y1 + crop_height
    
    # Handle wrapping for x coordinates (panorama wraps around)
    if x1 < 0 or x2 > pano_width:
        # Need to stitch from both ends
        view = Image.new('RGB', (crop_width, crop_height))
        
        if x1 < 0:
            # Left part from right edge of panorama
            left_width = -x1
            left_crop = panorama.crop((pano_width + x1, y1, pano_width, y2))
            view.paste(left_crop, (0, 0))
            
            # Right part from left edge
            right_crop = panorama.crop((0, y1, x2, y2))
            view.paste(right_crop, (left_width, 0))
        else:
            # Left part from right edge
            left_crop = panorama.crop((x1, y1, pano_width, y2))
            view.paste(left_crop, (0, 0))
            
            # Right part wraps around
            right_width = x2 - pano_width
            right_crop = panorama.crop((0, y1, right_width, y2))
            view.paste(right_crop, (pano_width - x1, 0))
    else:
        view = panorama.crop((x1, y1, x2, y2))
    
    # Resize to output size
    view = view.resize(output_size, Image.Resampling.LANCZOS)
    
    return view


# Simple test
if __name__ == "__main__":
    print("Testing panorama download...")
    
    # Test with a known panorama ID (Times Square, NYC)
    test_pano_id = "CAoSLEFGMVFpcE5fX0hRRVdCZGZJdnlOdXpZNkd2d0F5Y19PLXR3MXk5Z3FQSWM"
    
    img = download_full_panorama(test_pano_id, zoom=2)
    if img:
        img.save("test_panorama.jpg")
        print(f"✅ Saved test panorama: {img.size}")
        
        # Extract a view
        view = extract_view_from_panorama(img, heading=90, pitch=0, fov=90)
        view.save("test_view.jpg")
        print(f"✅ Saved test view: {view.size}")

