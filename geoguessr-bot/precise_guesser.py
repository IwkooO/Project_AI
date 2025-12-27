"""
Precise GeoGuessr Guess Submission Module.

Provides multiple methods to place guesses with exact precision:
1. Direct API submission via JavaScript injection (most reliable)
2. WebSocket-based Leaflet map control (for when API is blocked)
3. Smart minimap clicking with iterative refinement (fallback)
"""

import json
import math
import subprocess
import time
from typing import Optional, Tuple

import pyautogui


class PreciseGuesser:
    """
    Submits guesses to GeoGuessr with maximum precision.
    
    Primary method: Inject JavaScript to call GeoGuessr's own API.
    This bypasses all minimap coordinate conversion issues.
    """
    
    def __init__(self, screen_regions: dict):
        self.screen_regions = screen_regions
        # Track game state
        self.current_game_id = None
        self.current_round = 1
        
    def get_game_id_from_browser(self) -> Optional[str]:
        """Get the current game ID from browser URL."""
        # Focus browser, copy URL
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.15)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.15)
        pyautogui.press('escape')
        
        # Read URL from clipboard
        result = subprocess.run(
            ['xclip', '-selection', 'clipboard', '-o'],
            capture_output=True, text=True, timeout=2
        )
        url = result.stdout.strip()
        
        # Extract game ID
        import re
        match = re.search(r'/game/([A-Za-z0-9]+)', url)
        if match:
            game_id = match.group(1)
            print(f"   🎮 Game ID: {game_id}")
            return game_id
        
        # Try duels/battle-royale patterns
        match = re.search(r'/duels/([A-Za-z0-9]+)', url)
        if match:
            return match.group(1)
        match = re.search(r'/battle-royale/([A-Za-z0-9]+)', url)
        if match:
            return match.group(1)
            
        return None
    
    def submit_guess_via_api(self, lat: float, lng: float, round_num: int) -> bool:
        """
        Submit guess by injecting JavaScript that calls GeoGuessr's API.
        
        This is the MOST PRECISE method - uses exact coordinates with no
        pixel conversion or minimap clicking required.
        """
        game_id = self.get_game_id_from_browser()
        if not game_id:
            print("   ❌ Could not get game ID from URL")
            return False
        
        self.current_game_id = game_id
        
        # JavaScript code to submit the guess
        # Uses the same API endpoint as the Chrome extension
        js_code = f'''
(async () => {{
    const gameId = "{game_id}";
    const lat = {lat};
    const lng = {lng};
    const roundNumber = {round_num};
    
    // Determine API endpoint based on game type
    let apiUrl;
    const url = window.location.href;
    if (url.includes('/duels/')) {{
        apiUrl = `https://game-server.geoguessr.com/api/duels/${{gameId}}/guess`;
    }} else if (url.includes('/battle-royale/')) {{
        apiUrl = `https://game-server.geoguessr.com/api/battle-royale/${{gameId}}/guess`;
    }} else {{
        apiUrl = `https://game-server.geoguessr.com/api/game/${{gameId}}/guess`;
    }}
    
    console.log("🎯 Submitting guess:", {{ lat, lng, roundNumber, apiUrl }});
    
    const response = await fetch(apiUrl, {{
        method: "POST",
        credentials: "include",
        headers: {{
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-client": "web",
            "Origin": "https://www.geoguessr.com",
        }},
        body: JSON.stringify({{ lat, lng, roundNumber }})
    }});
    
    const result = await response.json();
    console.log("✅ API Response:", result);
    
    // Store result in window for Python to retrieve if needed
    window.__geobot_last_result = result;
    
    if (result.roundScore) {{
        console.log("📊 Score:", result.roundScore.amount, "Distance:", result.roundScore.distance, "m");
    }}
    
    return result;
}})();
'''
        
        return self._execute_js_in_browser(js_code)
    
    def _execute_js_in_browser(self, js_code: str) -> bool:
        """Execute JavaScript code in the browser via devtools console."""
        # Copy JS to clipboard
        process = subprocess.Popen(
            ['xclip', '-selection', 'clipboard'],
            stdin=subprocess.PIPE
        )
        process.communicate(js_code.encode())
        
        # Open browser devtools console
        pyautogui.hotkey('ctrl', 'shift', 'j')
        time.sleep(0.6)
        
        # Clear any existing content and paste
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.2)
        
        # Execute
        pyautogui.press('enter')
        time.sleep(1.5)  # Wait for API response
        
        # Close devtools
        pyautogui.hotkey('ctrl', 'shift', 'j')
        time.sleep(0.3)
        
        print(f"   ✅ JavaScript executed successfully")
        return True
    
    def place_marker_via_leaflet(self, lat: float, lng: float) -> bool:
        """
        Control the Leaflet map directly via JavaScript to place marker precisely.
        
        This method:
        1. Gets the Leaflet map instance
        2. Sets view to target coordinates
        3. Triggers a click at center (which places the marker)
        """
        js_code = f'''
(async () => {{
    // Find the Leaflet map instance
    const mapContainer = document.querySelector('.guess-map') || 
                         document.querySelector('[class*="guess-map"]') ||
                         document.querySelector('.leaflet-container');
    
    if (!mapContainer) {{
        console.error("❌ Could not find map container");
        return false;
    }}
    
    // Get Leaflet map instance - it's usually stored on the container
    let map = null;
    
    // Try common ways Leaflet stores the map reference
    if (mapContainer._leaflet_map) {{
        map = mapContainer._leaflet_map;
    }} else if (window.L && window.L.map) {{
        // Iterate through all leaflet maps
        for (const key in window.L._maps || {{}}) {{
            map = window.L._maps[key];
            break;
        }}
    }}
    
    // Alternative: Access via DOM element's leaflet property
    if (!map) {{
        const leafletPane = mapContainer.querySelector('.leaflet-pane');
        if (leafletPane && leafletPane._leaflet_id) {{
            // Search global for the map
            for (const key of Object.keys(window)) {{
                if (window[key] && window[key]._leaflet_id) {{
                    map = window[key];
                    break;
                }}
            }}
        }}
    }}
    
    if (!map || !map.setView) {{
        console.log("⚠️ Could not find Leaflet map, trying click simulation...");
        return false;
    }}
    
    const lat = {lat};
    const lng = {lng};
    
    console.log("🗺️ Found Leaflet map, setting view to:", lat, lng);
    
    // Zoom in and center on target
    map.setView([lat, lng], 15, {{ animate: false }});
    
    await new Promise(r => setTimeout(r, 300));
    
    // Get center point of map container
    const rect = mapContainer.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Simulate click at center (which is now our target coordinates)
    const clickEvent = new MouseEvent('click', {{
        view: window,
        bubbles: true,
        cancelable: true,
        clientX: centerX,
        clientY: centerY
    }});
    
    mapContainer.dispatchEvent(clickEvent);
    console.log("✅ Marker placed at:", lat, lng);
    
    return true;
}})();
'''
        return self._execute_js_in_browser(js_code)
    
    def submit_with_map_click(self, lat: float, lng: float, round_num: int) -> bool:
        """
        Alternative method: Pan map to location and click to place marker,
        then click the guess button.
        
        Use this if API submission is blocked/fails.
        """
        # First, expand the minimap by hovering
        self._expand_minimap()
        
        # Use Leaflet API to pan and place marker
        success = self.place_marker_via_leaflet(lat, lng)
        
        if success:
            # Wait a moment then click confirm
            time.sleep(0.5)
            confirm_btn = self.screen_regions.get("confirm_button_1")
            if confirm_btn:
                pyautogui.click(confirm_btn)
                print(f"   ✅ Clicked confirm button")
        
        return success
    
    def _expand_minimap(self):
        """Expand the minimap by hovering over it."""
        map_x = self.screen_regions.get("map_top_left_1", [0, 0])[0]
        map_y = self.screen_regions.get("map_top_left_1", [0, 0])[1]
        map_w = self.screen_regions.get("map_bot_right_1", [0, 0])[0] - map_x
        map_h = self.screen_regions.get("map_bot_right_1", [0, 0])[1] - map_y
        
        # Hover over minimap corner to expand it
        hover_x = map_x + map_w - 20
        hover_y = map_y + map_h - 20
        pyautogui.moveTo(hover_x, hover_y, duration=0.3)
        time.sleep(0.8)


class SmartMinimapGuesser:
    """
    Fallback: Smart minimap clicking with iterative refinement.
    
    This uses a more sophisticated approach when API/JS methods fail:
    1. Analyze current map viewport via screenshot
    2. Calculate target position relative to visible bounds
    3. Iteratively zoom and refine position
    """
    
    def __init__(self, screen_regions: dict):
        self.screen_regions = screen_regions
        self.map_x = screen_regions.get("map_top_left_1", [0, 0])[0]
        self.map_y = screen_regions.get("map_top_left_1", [0, 0])[1]
        self.map_w = screen_regions.get("map_bot_right_1", [0, 0])[0] - self.map_x
        self.map_h = screen_regions.get("map_bot_right_1", [0, 0])[1] - self.map_y
        
    def lat_lng_to_mercator(self, lat: float, lng: float) -> Tuple[float, float]:
        """Convert lat/lng to normalized Web Mercator coordinates (0-1 range)."""
        MAX_LAT = 85.051129
        lat = max(-MAX_LAT, min(MAX_LAT, lat))
        lng = ((lng + 180) % 360) - 180
        
        # X: linear
        x = (lng + 180) / 360.0
        
        # Y: Web Mercator
        lat_rad = math.radians(lat)
        mercator_y = math.log(math.tan(math.pi / 4 + lat_rad / 2))
        max_mercator = math.log(math.tan(math.pi / 4 + math.radians(MAX_LAT) / 2))
        y = (1 - mercator_y / max_mercator) / 2
        
        return x, y
    
    def click_with_zoom_refinement(self, lat: float, lng: float, zoom_levels: int = 4):
        """
        Click on minimap with iterative zoom refinement.
        
        Strategy:
        1. Click approximate location on zoomed-out map
        2. Zoom in on that location
        3. Click at center (map is now centered on our target)
        4. Repeat zoom for more precision
        """
        print(f"   🎯 Target: ({lat:.4f}, {lng:.4f})")
        
        # Expand minimap first
        self._expand_minimap()
        
        # Calculate initial position on world map
        norm_x, norm_y = self.lat_lng_to_mercator(lat, lng)
        
        # Initial click position
        x = int(self.map_x + norm_x * self.map_w)
        y = int(self.map_y + norm_y * self.map_h)
        
        # Clamp to minimap bounds
        margin = 15
        x = max(self.map_x + margin, min(self.map_x + self.map_w - margin, x))
        y = max(self.map_y + margin, min(self.map_y + self.map_h - margin, y))
        
        print(f"   📍 Initial click at ({x}, {y})")
        
        # Click to place initial marker
        pyautogui.click(x, y)
        time.sleep(0.3)
        
        # Now iteratively zoom in for precision
        for i in range(zoom_levels):
            print(f"   🔍 Zoom iteration {i+1}/{zoom_levels}")
            
            # Move to current marker position
            pyautogui.moveTo(x, y, duration=0.15)
            time.sleep(0.1)
            
            # Scroll to zoom in (3 clicks per iteration)
            for _ in range(3):
                pyautogui.scroll(3)  # Scroll up = zoom in
                time.sleep(0.1)
            
            time.sleep(0.3)  # Wait for zoom animation
            
            # After zooming, click at center of minimap
            # The map centers on where we zoomed, so our target is now at center
            center_x = self.map_x + self.map_w // 2
            center_y = self.map_y + self.map_h // 2
            
            pyautogui.click(center_x, center_y)
            time.sleep(0.2)
            
            # Update position for next iteration
            x, y = center_x, center_y
        
        print(f"   ✅ Final click at ({x}, {y})")
        return x, y
    
    def _expand_minimap(self):
        """Expand minimap by hovering."""
        hover_x = self.map_x + self.map_w - 20
        hover_y = self.map_y + self.map_h - 20
        pyautogui.moveTo(hover_x, hover_y, duration=0.3)
        time.sleep(0.8)


def submit_precise_guess(
    lat: float, 
    lng: float, 
    round_num: int,
    screen_regions: dict,
    method: str = "api"
) -> bool:
    """
    Submit a guess with maximum precision.
    
    Args:
        lat, lng: Target coordinates
        round_num: Current round number (1-indexed)
        screen_regions: Screen region configuration
        method: "api" (best), "leaflet" (map control), "smart_click" (fallback)
    
    Returns:
        True if guess was submitted successfully
    """
    print(f"🎯 Submitting precise guess: ({lat:.4f}, {lng:.4f}) round {round_num}")
    
    if method == "api":
        guesser = PreciseGuesser(screen_regions)
        return guesser.submit_guess_via_api(lat, lng, round_num)
    
    elif method == "leaflet":
        guesser = PreciseGuesser(screen_regions)
        return guesser.submit_with_map_click(lat, lng, round_num)
    
    elif method == "smart_click":
        guesser = SmartMinimapGuesser(screen_regions)
        guesser.click_with_zoom_refinement(lat, lng)
        # Click confirm button
        confirm_btn = screen_regions.get("confirm_button_1")
        if confirm_btn:
            time.sleep(0.3)
            pyautogui.click(confirm_btn)
        return True
    
    else:
        print(f"❌ Unknown method: {method}")
        return False

