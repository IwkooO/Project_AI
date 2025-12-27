"""
GeoGuessr Bot using PyAutoGUI and Stage 2 ML model API.

Takes screenshots, sends to API server, clicks on predicted location on minimap.
"""

import base64
import math
import os
import requests
from io import BytesIO
from time import sleep
from typing import Tuple, Optional

import pyautogui
from PIL import Image


class GeoBot:
    """Bot that plays GeoGuessr using ML model predictions."""
    
    def __init__(
        self, 
        screen_regions: dict, 
        player: int = 1,
        api_url: str = "http://127.0.0.1:5000/api/v1/predict"
    ):
        self.player = player
        self.screen_regions = screen_regions
        self.api_url = api_url
        
        # Screen region for capturing the panorama view
        self.screen_x, self.screen_y = screen_regions["screen_top_left"]
        self.screen_w = screen_regions["screen_bot_right"][0] - self.screen_x
        self.screen_h = screen_regions["screen_bot_right"][1] - self.screen_y
        self.screen_xywh = (self.screen_x, self.screen_y, self.screen_w, self.screen_h)

        # Minimap region
        self.map_x, self.map_y = screen_regions[f"map_top_left_{player}"]
        self.map_w = screen_regions[f"map_bot_right_{player}"][0] - self.map_x
        self.map_h = screen_regions[f"map_bot_right_{player}"][1] - self.map_y
        self.minimap_xywh = (self.map_x, self.map_y, self.map_w, self.map_h)

        # Button locations
        self.next_round_button = screen_regions.get("next_round_button")
        self.confirm_button = screen_regions[f"confirm_button_{player}"]
        
        print(f"🤖 GeoBot initialized")
        print(f"   📍 API: {api_url}")
        print(f"   📺 Screen region: {self.screen_w}x{self.screen_h}")
        print(f"   🗺️  Minimap region: {self.map_w}x{self.map_h}")

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict_location(self, image: Image.Image) -> Optional[Tuple[float, float]]:
        """Send screenshot to ML API and return predicted lat/lng."""
        image_b64 = self.pil_to_base64(image)
        payload = {"image": f"data:image/png;base64,{image_b64}"}
        
        response = requests.post(
            self.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        lat = result["results"]["lat"]
        lng = result["results"]["lng"]
        
        return lat, lng

    def lat_lon_to_screen_coords(self, lat: float, lng: float) -> Tuple[int, int]:
        """
        Convert latitude and longitude to pixel coordinates on the minimap.
        Uses Web Mercator projection - maps world to minimap bounds.
        """
        # Clamp latitude to valid Mercator range
        lat = max(-85, min(85, lat))
        
        # X: Linear mapping of longitude (-180 to 180) to minimap width
        x_ratio = (lng + 180) / 360.0
        x = self.map_x + int(x_ratio * self.map_w)
        
        # Y: Mercator projection for latitude
        # Convert lat to radians
        lat_rad = math.radians(lat)
        # Mercator Y formula (normalized to 0-1 range)
        mercator_y = (1 - (math.log(math.tan(lat_rad) + 1/math.cos(lat_rad)) / math.pi)) / 2
        y = self.map_y + int(mercator_y * self.map_h)
        
        print(f"   Conversion: ({lat:.2f}, {lng:.2f}) -> ratio ({x_ratio:.3f}, {mercator_y:.3f}) -> pixel ({x}, {y})")
        
        return x, y

    def clamp_to_minimap(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to be within the minimap bounds."""
        margin = 10
        x = max(self.map_x + margin, min(self.map_x + self.map_w - margin, x))
        y = max(self.map_y + margin, min(self.map_y + self.map_h - margin, y))
        return x, y

    def expand_minimap(self):
        """Hover over minimap to expand it."""
        # Move to bottom-right corner of minimap to trigger expansion
        hover_x = self.map_x + self.map_w - 20
        hover_y = self.map_y + self.map_h - 20
        pyautogui.moveTo(hover_x, hover_y, duration=0.3)
        sleep(0.8)  # Wait for expansion animation

    def click_on_map(self, x: int, y: int):
        """Click on the minimap at the specified pixel location."""
        pyautogui.click(x, y)
        sleep(0.3)

    def click_confirm(self):
        """Click the confirm/guess button."""
        pyautogui.click(self.confirm_button)
        sleep(0.5)

    def next_round(self):
        """Advance to the next round."""
        sleep(2)  # Wait for results to show
        
        if self.next_round_button:
            pyautogui.click(self.next_round_button)
        else:
            # Press space to continue
            pyautogui.press("space")
        
        sleep(2)  # Wait for next round to load


def play_round(bot: GeoBot, round_num: int, save_screenshots: bool = True) -> bool:
    """
    Play a single round of GeoGuessr.
    
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*50}")
    print(f"🎮 ROUND {round_num}")
    print(f"{'='*50}")
    
    # Wait for panorama to load
    print("⏳ Waiting for panorama to load...")
    sleep(2)
    
    # Take screenshot of panorama
    print("📸 Taking screenshot...")
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)
    
    if save_screenshots:
        os.makedirs("screenshots", exist_ok=True)
        screenshot.save(f"screenshots/round_{round_num}.png")
        print(f"   Saved to screenshots/round_{round_num}.png")
    
    # Get prediction from ML model
    print("🔮 Getting ML model prediction...")
    result = bot.predict_location(screenshot)
    
    if result is None:
        print("❌ Failed to get prediction!")
        # Fallback: click center of map
        x = bot.map_x + bot.map_w // 2
        y = bot.map_y + bot.map_h // 2
        print(f"   Using fallback: center of map ({x}, {y})")
    else:
        lat, lng = result
        print(f"📍 Predicted: {lat:.4f}, {lng:.4f}")
        
        # Convert to screen coordinates
        x, y = bot.lat_lon_to_screen_coords(lat, lng)
        print(f"🖱️  Screen coords: ({x}, {y})")
        
        # Clamp to minimap bounds
        x, y = bot.clamp_to_minimap(x, y)
        print(f"   Clamped to: ({x}, {y})")
    
    # Expand minimap
    print("🗺️  Expanding minimap...")
    bot.expand_minimap()
    
    # Click on predicted location
    print("🖱️  Clicking on map...")
    bot.click_on_map(x, y)
    
    # Click confirm button
    print("✅ Confirming guess...")
    bot.click_confirm()
    
    # Move to next round
    print("⏭️  Moving to next round...")
    bot.next_round()
    
    print(f"✅ Round {round_num} complete!")
    return True


def play_game(bot: GeoBot, num_rounds: int = 5, save_screenshots: bool = True):
    """Play a full game of GeoGuessr."""
    print("\n" + "="*60)
    print("🎮 GEOGUESSR ML BOT - STARTING GAME")
    print("="*60)
    print(f"   Rounds to play: {num_rounds}")
    print(f"   API endpoint: {bot.api_url}")
    print("="*60)
    
    for round_num in range(1, num_rounds + 1):
        success = play_round(bot, round_num, save_screenshots)
        if not success:
            print(f"⚠️ Round {round_num} had issues, continuing...")
    
    print("\n" + "="*60)
    print("🎉 GAME COMPLETE!")
    print("="*60)
