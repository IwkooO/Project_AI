"""
GeoGuessr Bot using PyAutoGUI and our Stage 2 ML model API.

Takes screenshots, sends to API server, clicks on predicted location.
"""

import base64
import math
import os
import requests
from io import BytesIO
from time import sleep
from typing import Tuple, List, Optional

import pyautogui
import matplotlib.pyplot as plt
from PIL import Image


API_ENDPOINT = "http://127.0.0.1:5000/api/v1/predict"


class GeoBot:
    def __init__(self, screen_regions: dict, player: int = 1):
        self.player = player
        self.screen_regions = screen_regions
        
        # Screen region for capturing the game view
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
        self.next_round_button = screen_regions.get("next_round_button") if player == 1 else None
        self.confirm_button = screen_regions[f"confirm_button_{player}"]

        # Reference points for coordinate conversion (Kodiak, Alaska and Hobart, Tasmania)
        self.kodiak_x, self.kodiak_y = screen_regions[f"kodiak_{player}"]
        self.hobart_x, self.hobart_y = screen_regions[f"hobart_{player}"]
        
        # Known lat/lon of reference points
        self.kodiak_lat, self.kodiak_lon = (57.7916, -152.4083)
        self.hobart_lat, self.hobart_lon = (-42.8833, 147.3355)

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64_str

    def call_api(self, image_b64: str) -> Optional[Tuple[float, float]]:
        """Call the ML model API and return predicted lat/lng."""
        payload = {"image": f"data:image/png;base64,{image_b64}"}
        
        response = requests.post(
            API_ENDPOINT,
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

    @staticmethod
    def lat_to_mercator_y(lat: float) -> float:
        """Convert latitude to Mercator Y coordinate."""
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def lat_lon_to_mercator_map_pixels(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert latitude and longitude to pixel coordinates on the mercator projection minimap.
        Uses two known reference points (Kodiak and Hobart) for calibration.
        """
        # Calculate the x pixel coordinate
        lon_diff_ref = (self.kodiak_lon - self.hobart_lon)
        lon_diff = (self.kodiak_lon - lon)
        x = abs(self.kodiak_x - self.hobart_x) * (lon_diff / lon_diff_ref) + self.kodiak_x

        # Convert to mercator projection y coordinates
        mercator_y1 = self.lat_to_mercator_y(self.kodiak_lat)
        mercator_y2 = self.lat_to_mercator_y(self.hobart_lat)
        mercator_y = self.lat_to_mercator_y(lat)

        # Calculate the y pixel coordinate
        lat_diff_ref = (mercator_y1 - mercator_y2)
        lat_diff = (mercator_y1 - mercator_y)
        y = abs(self.kodiak_y - self.hobart_y) * (lat_diff / lat_diff_ref) + self.kodiak_y

        return round(x), round(y)

    def clamp_to_minimap(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to be within the minimap bounds."""
        if x < self.map_x:
            x = self.map_x + 5
            print("⚠️ x clamped to left bound")
        elif x > self.map_x + self.map_w:
            x = self.map_x + self.map_w - 5
            print("⚠️ x clamped to right bound")
        
        if y < self.map_y:
            y = self.map_y + 5
            print("⚠️ y clamped to top bound")
        elif y > self.map_y + self.map_h:
            y = self.map_y + self.map_h - 5
            print("⚠️ y clamped to bottom bound")
        
        return x, y

    def select_map_location(self, x: int, y: int, plot: bool = False) -> None:
        """Click on the minimap at the specified pixel location and confirm."""
        # Hover over minimap to expand it
        pyautogui.moveTo(self.map_x + self.map_w - 15, self.map_y + self.map_h - 15, duration=0.3)
        sleep(0.5)

        # Click on the predicted location
        pyautogui.click(x, y, duration=0.3)
        sleep(0.3)

        if plot:
            self.plot_minimap(x, y)

        # Confirm the guess
        pyautogui.click(self.confirm_button, duration=0.2)
        sleep(2)

    def plot_minimap(self, x: int = None, y: int = None) -> None:
        """Save a plot of the minimap with reference points and prediction."""
        minimap = pyautogui.screenshot(region=self.minimap_xywh)
        
        plot_kodiak_x = self.kodiak_x - self.map_x
        plot_kodiak_y = self.kodiak_y - self.map_y
        plot_hobart_x = self.hobart_x - self.map_x
        plot_hobart_y = self.hobart_y - self.map_y
        
        plt.figure(figsize=(10, 6))
        plt.imshow(minimap)
        plt.plot(plot_hobart_x, plot_hobart_y, 'ro', markersize=8, label='Hobart (ref)')
        plt.plot(plot_kodiak_x, plot_kodiak_y, 'go', markersize=8, label='Kodiak (ref)')
        
        if x and y:
            plt.plot(x - self.map_x, y - self.map_y, 'b*', markersize=15, label='Prediction')
        
        plt.legend()
        plt.title("Minimap with prediction")
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/minimap.png")
        plt.close()
        print("📊 Minimap plot saved to plots/minimap.png")


def play_turn(bot: GeoBot, turn: int, plot: bool = False) -> bool:
    """Play a single turn of GeoGuessr."""
    print(f"\n{'='*50}")
    print(f"🎮 Turn {turn}")
    print(f"{'='*50}")
    
    # Wait a moment for the panorama to load
    sleep(1)
    
    # Take screenshot
    print("📸 Taking screenshot...")
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)
    screenshot_b64 = GeoBot.pil_to_base64(screenshot)
    
    # Save screenshot for debugging
    os.makedirs("plots", exist_ok=True)
    screenshot.save(f"plots/screenshot_turn_{turn}.png")
    print(f"📸 Screenshot saved to plots/screenshot_turn_{turn}.png")
    
    # Call API
    print("🔮 Calling ML model API...")
    result = bot.call_api(screenshot_b64)
    
    if result is None:
        print("❌ Failed to get prediction from API")
        # Click somewhere random on the map as fallback
        bot.select_map_location(bot.map_x + bot.map_w // 2, bot.map_y + bot.map_h // 2, plot=plot)
        return False
    
    lat, lng = result
    print(f"📍 Predicted location: {lat:.4f}, {lng:.4f}")
    
    # Convert to pixel coordinates
    x, y = bot.lat_lon_to_mercator_map_pixels(lat, lng)
    print(f"🖱️ Pixel coordinates: ({x}, {y})")
    
    # Clamp to minimap bounds
    x, y = bot.clamp_to_minimap(x, y)
    
    # Select location on map
    print("🗺️ Selecting location on map...")
    bot.select_map_location(x, y, plot=plot)
    
    # Go to next round (press space or wait)
    print("⏭️ Moving to next round...")
    sleep(1)
    pyautogui.press("space")
    sleep(2)
    
    return True

