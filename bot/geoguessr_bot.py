"""
GeoGuessr Bot using PyAutoGUI and Stage 2 ML model API.

Takes screenshots, sends to API server, clicks on predicted location on minimap.
Concept logging happens server-side automatically.
Results tracking captures true locations and scores.
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

from pyautogui_tracker import PyAutoGUIResultsTracker as ResultsTracker


class GeoBot:
    """Bot that plays GeoGuessr using ML model predictions."""
    
    # Calibration values for mapping coordinates to minimap pixels
    # These account for the fact that GeoGuessr's minimap doesn't show
    # exactly -180 to 180 longitude / -85 to 85 latitude
    # Adjust these if guesses are consistently off in one direction
    MAP_CALIBRATION = {
        "x_offset": 0.01,   # Horizontal offset (0 = no offset)
        "y_offset": 0.03,   # Vertical offset (0 = no offset)  
        "x_scale": 0.98,    # Horizontal scale (1 = full range)
        "y_scale": 0.94,    # Vertical scale (1 = full range)
    }
    
    def __init__(
        self, 
        screen_regions: dict, 
        player: int = 1,
        api_url: str = "http://127.0.0.1:5000/api/v1/predict",
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
        print(f"   📐 Calibration: {self.MAP_CALIBRATION}")

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

    def lat_lng_to_mercator(self, lat: float, lng: float) -> Tuple[float, float]:
        """
        Convert lat/lng to normalized Web Mercator coordinates [0,1].
        
        Returns (x_ratio, y_ratio) where:
        - x_ratio: 0 = -180° longitude, 1 = +180° longitude
        - y_ratio: 0 = north (85°), 1 = south (-85°)
        """
        MAX_LAT = 85.051129
        lat = max(-MAX_LAT, min(MAX_LAT, lat))
        lng = ((lng + 180) % 360) - 180
        
        # X: Linear mapping of longitude
        x_ratio = (lng + 180) / 360.0
        
        # Y: Web Mercator projection
        lat_rad = math.radians(lat)
        mercator_y = math.log(math.tan(math.pi / 4 + lat_rad / 2))
        max_mercator = math.log(math.tan(math.pi / 4 + math.radians(MAX_LAT) / 2))
        y_ratio = (1 - mercator_y / max_mercator) / 2
        
        return x_ratio, y_ratio
    
    def mercator_to_screen(self, x_ratio: float, y_ratio: float, 
                           map_bounds: Tuple[float, float, float, float] = None) -> Tuple[int, int]:
        """
        Convert normalized mercator ratios to screen pixel coordinates.
        
        Args:
            x_ratio, y_ratio: Normalized [0,1] mercator coordinates
            map_bounds: (min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio) of visible map
                       If None, assumes full world view (0,0,1,1)
        """
        if map_bounds is None:
            map_bounds = (0.0, 0.0, 1.0, 1.0)
        
        min_x, min_y, max_x, max_y = map_bounds
        
        # Calculate position within visible bounds
        if max_x > min_x:
            pixel_x_ratio = (x_ratio - min_x) / (max_x - min_x)
        else:
            pixel_x_ratio = 0.5
            
        if max_y > min_y:
            pixel_y_ratio = (y_ratio - min_y) / (max_y - min_y)
        else:
            pixel_y_ratio = 0.5
        
        # Map to screen coordinates
        x = self.map_x + int(pixel_x_ratio * self.map_w)
        y = self.map_y + int(pixel_y_ratio * self.map_h)
        
        return x, y

    def lat_lon_to_screen_coords(self, lat: float, lng: float) -> Tuple[int, int]:
        """
        Convert latitude and longitude to pixel coordinates on the minimap.
        Assumes full world view (zoomed out completely).
        """
        x_ratio, y_ratio = self.lat_lng_to_mercator(lat, lng)
        x, y = self.mercator_to_screen(x_ratio, y_ratio)
        
        print(f"   Conversion: ({lat:.4f}, {lng:.4f}) -> pixel ({x}, {y})")
        print(f"   Ratios: x={x_ratio:.4f}, y={y_ratio:.4f}")
        
        return x, y

    def clamp_to_minimap(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to be within the minimap bounds."""
        margin = 10
        x_clamped = max(self.map_x + margin, min(self.map_x + self.map_w - margin, x))
        y_clamped = max(self.map_y + margin, min(self.map_y + self.map_h - margin, y))
        
        if x != x_clamped or y != y_clamped:
            print(f"   ⚠️ Clamped from ({x}, {y}) to ({x_clamped}, {y_clamped})")
        
        return x_clamped, y_clamped

    def expand_minimap(self):
        """Hover over minimap to expand it."""
        hover_x = self.map_x + self.map_w - 20
        hover_y = self.map_y + self.map_h - 20
        pyautogui.moveTo(hover_x, hover_y, duration=0.3)
        sleep(0.8)

    def reset_minimap_zoom(self):
        """
        Zoom out the minimap fully to reset to world view.
        This gives us a known state to work from.
        """
        center_x = self.map_x + self.map_w // 2
        center_y = self.map_y + self.map_h // 2
        
        pyautogui.moveTo(center_x, center_y, duration=0.1)
        sleep(0.1)
        
        # Zoom out a lot to ensure we're at world view
        for _ in range(12):
            pyautogui.scroll(-5)  # Negative = zoom out
            sleep(0.05)
        
        sleep(0.4)
        print("   🌍 Reset minimap to world view")
    
    def smart_place_guess(self, lat: float, lng: float, debug: bool = True):
        """
        Precisely place a guess on the GeoGuessr minimap using iterative zoom-and-center.
        
        Strategy (using double-click centering - most reliable):
        1. Expand minimap
        2. Zoom out fully to get world view (known state)
        3. Calculate where target is on world view
        4. Double-click to zoom in AND center the view on that point
        5. Repeat double-click zoom to increase precision
        6. Single click at center to place marker
        
        Double-click zoom is more reliable than scroll-zoom because:
        - It ALWAYS centers the view on the clicked point
        - It works consistently across all map implementations
        """
        print(f"   🎯 Smart placement for ({lat:.4f}, {lng:.4f})")
        
        # Step 1: Ensure minimap is expanded
        self.expand_minimap()
        
        # Step 2: Reset to world view (fully zoomed out)
        self.reset_minimap_zoom()
        
        # Get minimap center coordinates
        center_x = self.map_x + self.map_w // 2
        center_y = self.map_y + self.map_h // 2
        
        # Step 3: Calculate target position on world view
        target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
        
        # Apply calibration from class constants
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
        
        # Clamp ratios to valid range
        adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
        adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
        
        # Convert to screen coordinates
        target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
        target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   📍 World view position: ({target_x}, {target_y})")
        print(f"   📐 Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
        
        if debug:
            # Save debug screenshot showing where we intend to click
            from PIL import ImageDraw
            debug_img = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
            draw = ImageDraw.Draw(debug_img)
            local_x = target_x - self.map_x
            local_y = target_y - self.map_y
            draw.ellipse([local_x-10, local_y-10, local_x+10, local_y+10], outline="red", width=3)
            draw.line([local_x-15, local_y, local_x+15, local_y], fill="red", width=2)
            draw.line([local_x, local_y-15, local_x, local_y+15], fill="red", width=2)
            debug_img.save("debug_target_position.png")
            print(f"   📸 Debug image saved to debug_target_position.png")
        
        # Step 4: Use iterative double-click zoom to center on target
        # Each double-click zooms in AND centers the view on that point
        
        # First double-click: zoom in on approximate target
        print("   🔍 Double-click zoom iteration 1...")
        pyautogui.moveTo(target_x, target_y, duration=0.15)
        sleep(0.1)
        pyautogui.doubleClick(target_x, target_y)
        sleep(0.5)
        
        # After first zoom, the target should now be near center
        # Do a second double-click at center to zoom more
        print("   🔍 Double-click zoom iteration 2...")
        pyautogui.doubleClick(center_x, center_y)
        sleep(0.5)
        
        # Third zoom for high precision
        print("   🔍 Double-click zoom iteration 3...")
        pyautogui.doubleClick(center_x, center_y)
        sleep(0.5)
        
        # Step 5: Final single click at center to place marker
        # After 3 double-clicks, we're zoomed in and centered on target
        print(f"   ✅ Placing marker at center ({center_x}, {center_y})")
        pyautogui.click(center_x, center_y)
        
        sleep(0.2)

    def scroll_zoom_place_guess(self, lat: float, lng: float, debug: bool = True):
        """
        Primary method using scroll zoom (keeps point under cursor fixed).
        
        This is more reliable than double-click because:
        - Scroll zoom keeps the point under cursor stationary
        - Works consistently across Leaflet/Google Maps implementations
        - Less prone to centering errors
        """
        print(f"   🎯 Scroll-zoom placement for ({lat:.4f}, {lng:.4f})")
        
        # Step 1: Expand minimap
        self.expand_minimap()
        
        # Step 2: Reset to world view
        self.reset_minimap_zoom()
        
        # Step 3: Calculate target position using calibration
        target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
        
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
        adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
        adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
        
        target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
        target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   📍 Target position: ({target_x}, {target_y})")
        print(f"   📐 Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
        
        if debug:
            # Save debug screenshot showing calculated target
            from PIL import ImageDraw
            debug_img = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
            draw = ImageDraw.Draw(debug_img)
            local_x = target_x - self.map_x
            local_y = target_y - self.map_y
            # Draw crosshair at target
            draw.ellipse([local_x-10, local_y-10, local_x+10, local_y+10], outline="lime", width=3)
            draw.line([local_x-15, local_y, local_x+15, local_y], fill="lime", width=2)
            draw.line([local_x, local_y-15, local_x, local_y+15], fill="lime", width=2)
            # Add text with coordinates
            debug_img.save("debug_scroll_target.png")
            print(f"   📸 Debug image saved to debug_scroll_target.png")
        
        # Step 4: Move to target and scroll zoom
        # Scroll zoom keeps the point under cursor fixed
        pyautogui.moveTo(target_x, target_y, duration=0.2)
        sleep(0.15)
        
        print("   🔍 Scroll zooming on target (8 clicks)...")
        for i in range(8):
            pyautogui.scroll(5)  # Zoom in
            sleep(0.08)
        sleep(0.4)
        
        # Step 5: Click at current cursor position (target should still be here)
        current_x, current_y = pyautogui.position()
        print(f"   ✅ Clicking at cursor position ({current_x}, {current_y})")
        pyautogui.click(current_x, current_y)
        
        sleep(0.2)
    
    def zoom_minimap(self, x: int, y: int, zoom_clicks: int = 3):
        """
        Zoom into the minimap at the target location for more precise clicking.
        
        Args:
            x, y: Target screen coordinates to zoom towards
            zoom_clicks: Number of scroll wheel clicks to zoom in
        """
        # Move to target location on minimap
        pyautogui.moveTo(x, y, duration=0.2)
        sleep(0.2)
        
        # Scroll to zoom in (positive = zoom in on most maps)
        for _ in range(zoom_clicks):
            pyautogui.scroll(3)  # Scroll up to zoom in
            sleep(0.15)
        
        sleep(0.3)  # Wait for zoom animation
        
    def click_on_map_precise(self, lat: float, lng: float, use_zoom: bool = True, method: str = "scroll"):
        """
        Click on the minimap with zoom-based precision.
        
        Args:
            lat, lng: Target coordinates
            use_zoom: Whether to use zoom for better precision
            method: "scroll" (scroll zoom, keeps cursor on target) or "doubleclick" (double-click to center)
        """
        if use_zoom:
            if method == "doubleclick":
                self.smart_place_guess(lat, lng)
            else:
                # Default: scroll zoom method - more reliable for most maps
                self.scroll_zoom_place_guess(lat, lng)
        else:
            x, y = self.lat_lon_to_screen_coords(lat, lng)
            x, y = self.clamp_to_minimap(x, y)
            pyautogui.click(x, y)
            print(f"   🎯 Clicked at ({x}, {y})")
        
        sleep(0.3)
    
    def calibration_test(self, test_coords: list = None):
        """
        Test the coordinate placement accuracy with known locations.
        
        Run this manually to verify calibration is correct.
        Adjust x_offset, y_offset, x_scale, y_scale in smart_place_guess if needed.
        """
        if test_coords is None:
            # Test with well-known locations
            test_coords = [
                (0, 0, "Gulf of Guinea (0,0)"),
                (48.8566, 2.3522, "Paris"),
                (40.7128, -74.0060, "New York"),
                (-33.8688, 151.2093, "Sydney"),
                (35.6762, 139.6503, "Tokyo"),
                (-22.9068, -43.1729, "Rio de Janeiro"),
            ]
        
        print("\n" + "="*50)
        print("🧪 CALIBRATION TEST")
        print("="*50)
        print("Watch where the cursor moves - it should match the location names")
        print("Press Ctrl+C to stop")
        print("="*50)
        
        for lat, lng, name in test_coords:
            print(f"\n📍 Testing: {name} ({lat:.4f}, {lng:.4f})")
            
            # Expand and reset
            self.expand_minimap()
            self.reset_minimap_zoom()
            
            # Calculate position using class calibration
            target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
            
            cal = self.MAP_CALIBRATION
            adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
            adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
            adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
            adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
            
            target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
            target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
            
            print(f"   Calculated pixel: ({target_x}, {target_y})")
            print(f"   Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
            
            # Move cursor to show where we would click
            pyautogui.moveTo(target_x, target_y, duration=0.5)
            sleep(2)  # Pause to observe

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
        sleep(2)
        
        if self.next_round_button:
            pyautogui.click(self.next_round_button)
        else:
            pyautogui.press("space")
        
        sleep(2)


def play_round(
    bot: GeoBot,
    round_num: int,
    save_screenshots: bool = True,
    tracker: Optional[ResultsTracker] = None
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Play a single round of GeoGuessr.
    
    Returns:
        Tuple of (success, (pred_lat, pred_lng) or None)
    """
    print(f"\n{'='*50}")
    print(f"🎮 ROUND {round_num}")
    print(f"{'='*50}")
    
    pred_lat, pred_lng = None, None
    
    # Wait for panorama to load
    print("⏳ Waiting for panorama to load...")
    sleep(2)

    # Panorama ID extraction disabled - not needed for basic bot operation
    # The true location will be extracted from the results screen marker

    # Take screenshot
    print("📸 Taking screenshot...")
    image = pyautogui.screenshot(region=bot.screen_xywh)
    
    if save_screenshots:
        os.makedirs("screenshots", exist_ok=True)
        image.save(f"screenshots/round_{round_num}.png")
        print(f"   Saved to screenshots/round_{round_num}.png")
    
    # Get prediction from ML model (server also logs concepts)
    print("🔮 Getting ML model prediction...")
    result = bot.predict_location(image)
    
    if result is None:
        print("❌ Failed to get prediction!")
        x = bot.map_x + bot.map_w // 2
        y = bot.map_y + bot.map_h // 2
        print(f"   Using fallback: center of map ({x}, {y})")
        use_precision = False
    else:
        pred_lat, pred_lng = result
        print(f"📍 Predicted: {pred_lat:.4f}, {pred_lng:.4f}")
        
        # Record prediction if tracker is active
        if tracker:
            tracker.record_prediction(round_num, pred_lat, pred_lng)
        
        # Convert to screen coordinates (for fallback)
        x, y = bot.lat_lon_to_screen_coords(pred_lat, pred_lng)
        
        # Clamp to minimap bounds
        x, y = bot.clamp_to_minimap(x, y)
        use_precision = True
    
    # Click on predicted location with precision zoom
    # (smart_place_guess handles minimap expansion internally)
    print("🖱️  Placing guess on map...")
    if use_precision and pred_lat is not None:
        bot.click_on_map_precise(pred_lat, pred_lng, use_zoom=True)
    else:
        bot.expand_minimap()
        bot.click_on_map(x, y)
    
    # Click confirm button
    print("✅ Confirming guess...")
    bot.click_confirm()
    
    # Wait for results screen to fully load
    sleep(3)
    
    # Capture round result (true location + score) if tracker is active
    if tracker and pred_lat is not None:
        tracker.capture_round_result(round_num, bot.screen_regions)
    
    # Move to next round
    print("⏭️  Moving to next round...")
    bot.next_round()
    
    print(f"✅ Round {round_num} complete!")
    return True, (pred_lat, pred_lng) if pred_lat is not None else None


def play_game(
    bot: GeoBot,
    num_rounds: int = 5,
    save_screenshots: bool = True,
    track_results: bool = True,
    results_output_dir: str = "results"
):
    """
    Play a full game of GeoGuessr.
    
    Args:
        bot: GeoBot instance
        num_rounds: Number of rounds to play
        save_screenshots: Whether to save screenshots locally
        track_results: Whether to track true locations and scores
        results_output_dir: Directory to save results CSV
    """
    print("\n" + "="*60)
    print("🎮 GEOGUESSR ML BOT - STARTING GAME")
    print("="*60)
    print(f"   Rounds to play: {num_rounds}")
    print(f"   API endpoint: {bot.api_url}")
    print(f"   Track results: {track_results}")
    print("="*60)
    
    tracker = None
    if track_results:
        tracker = ResultsTracker(output_dir=results_output_dir)
        tracker.connect()  # Always succeeds for PyAutoGUI tracker
    
    for round_num in range(1, num_rounds + 1):
        success, prediction = play_round(bot, round_num, save_screenshots, tracker)
        if not success:
            print(f"⚠️ Round {round_num} had issues, continuing...")
    
    # Print summary
    if tracker:
        tracker.print_summary()
        tracker.close()
    
    print("\n" + "="*60)
    print("🎉 GAME COMPLETE!")
    print("="*60)
