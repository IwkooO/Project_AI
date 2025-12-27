"""
Results Tracker for GeoGuessr Bot.

Captures true location and score by intercepting GeoGuessr API responses.
Uses Selenium with Chrome DevTools Protocol for network interception.
"""

import csv
import json
import math
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


@dataclass
class RoundResult:
    """Results for a single round."""
    round_num: int
    predicted_lat: float
    predicted_lng: float
    true_lat: Optional[float] = None
    true_lng: Optional[float] = None
    distance_km: Optional[float] = None
    score: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_geoguessr_score(distance_km: float, max_distance_km: float = 14916.862) -> int:
    """
    Calculate GeoGuessr-style score using the non-linear decay formula.
    
    Score = 5000 * exp(-10 * (d / D))
    
    Where:
        d = distance between guess and true location (km)
        D = max distance (map diagonal, default is world map ~14917 km)
    
    Args:
        distance_km: Distance between guess and true location in km
        max_distance_km: Maximum possible distance (map diagonal). 
                         Default 14916.862 km is Earth's half circumference (world map).
                         For country maps, use smaller values.
    
    Returns:
        Score between 0 and 5000
    """
    if distance_km <= 0:
        return 5000
    
    score = 5000 * math.exp(-10 * (distance_km / max_distance_km))
    return int(round(score))


class ResultsTracker:
    """
    Track GeoGuessr results by intercepting network traffic.
    
    Requires Chrome to be running with remote debugging enabled:
    google-chrome --remote-debugging-port=9222
    """
    
    def __init__(self, output_dir: str = "results", chrome_debug_port: int = 9222, stage1_checkpoint: str = None, stage2_checkpoint: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chrome_debug_port = chrome_debug_port
        self.driver: Optional[webdriver.Chrome] = None
        self.results: List[RoundResult] = []

        # Checkpoint information
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint

        # Session timestamp for CSV filename
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"game_results_{self.session_timestamp}.csv"

        # Initialize CSV with headers
        self._init_csv()

        print(f"📊 Results tracker initialized")
        print(f"   Output: {self.csv_path}")
        if self.stage1_checkpoint:
            print(f"   Stage1 checkpoint: {self.stage1_checkpoint}")
        if self.stage2_checkpoint:
            print(f"   Stage2 checkpoint: {self.stage2_checkpoint}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp',
                'predicted_lat', 'predicted_lng',
                'true_lat', 'true_lng',
                'distance_km', 'score',
                'stage1_checkpoint', 'stage2_checkpoint'
            ])
    
    def connect_to_chrome(self) -> bool:
        """Connect to Chrome with remote debugging."""
        options = Options()
        options.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.chrome_debug_port}")
        
        # Enable performance logging to capture network events
        options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
        
        self.driver = webdriver.Chrome(options=options)
        print(f"✅ Connected to Chrome on port {self.chrome_debug_port}")
        return True
    
    def get_latest_guess_response(self) -> Optional[dict]:
        """
        Parse Chrome DevTools logs to find the latest guess API response.
        
        GeoGuessr guess endpoint returns data like:
        {
            "lat": <true_lat>,
            "lng": <true_lng>,
            "distance": {"meters": {"amount": X}},
            "roundScore": {"amount": X}
        }
        """
        if self.driver is None:
            return None
        
        logs = self.driver.get_log('performance')
        
        # Look for guess response (search from most recent)
        for entry in reversed(logs):
            try:
                message = json.loads(entry['message'])['message']
                
                if message['method'] == 'Network.responseReceived':
                    url = message['params']['response']['url']
                    
                    # GeoGuessr guess endpoints
                    if '/api/game/' in url and '/guess' in url:
                        request_id = message['params']['requestId']
                        
                        # Get response body
                        try:
                            body = self.driver.execute_cdp_cmd(
                                'Network.getResponseBody',
                                {'requestId': request_id}
                            )
                            return json.loads(body['body'])
                        except Exception:
                            continue
            except (json.JSONDecodeError, KeyError):
                continue
        
        return None
    
    def record_prediction(self, round_num: int, pred_lat: float, pred_lng: float):
        """Record a prediction (called when bot makes a guess)."""
        result = RoundResult(
            round_num=round_num,
            predicted_lat=pred_lat,
            predicted_lng=pred_lng
        )
        self.results.append(result)
        print(f"📝 Recorded prediction for round {round_num}: ({pred_lat:.4f}, {pred_lng:.4f})")
    
    def capture_round_result(self, round_num: int) -> Optional[RoundResult]:
        """
        Capture the true location by clicking the correct location marker.
        
        The marker opens a Google Maps link with coordinates in viewpoint= parameter.
        """
        if round_num > len(self.results):
            print(f"⚠️ No prediction recorded for round {round_num}")
            return None
        
        result = self.results[round_num - 1]
        
        if not self.driver:
            print(f"⚠️ No browser connection for round {round_num}")
            self._append_to_csv(result)
            return result
        
        # Click the marker and extract coordinates from Google Maps URL
        coords = self._click_marker_and_get_coords()
        
        if coords:
            result.true_lat = coords[0]
            result.true_lng = coords[1]
            
            # Calculate distance
            result.distance_km = haversine_distance(
                result.predicted_lat, result.predicted_lng,
                result.true_lat, result.true_lng
            )

            # Calculate score
            result.score = calculate_geoguessr_score(result.distance_km)
            
            print(f"📊 Round {round_num} Results:")
            print(f"   True location: ({result.true_lat:.4f}, {result.true_lng:.4f})")
            print(f"   Distance: {result.distance_km:.1f} km")
            print(f"   Score: {result.score}")
        else:
            print(f"⚠️ Could not extract results for round {round_num}")
        
        # Write to CSV
        self._append_to_csv(result)
        
        return result

    def _click_marker_and_get_coords(self) -> Optional[tuple]:
        """
        Click the correct location marker and extract coordinates from Google Maps URL.
        
        Returns (lat, lng) tuple or None if extraction fails.
        """
        original_window = None
        try:
            original_window = self.driver.current_window_handle
            original_windows = set(self.driver.window_handles)
            
            # Wait for the marker to appear (results screen loads)
            print("   ⏳ Looking for correct location marker...")
            marker = None
            for _ in range(40):  # Wait up to 8 seconds
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-qa="correct-location-marker"]')
                    for el in elements:
                        if el.is_displayed():
                            marker = el
                            break
                    if marker:
                        break
                except:
                    pass
                time.sleep(0.2)
            
            if not marker:
                print("   ❌ Marker not found")
                return None
            
            print("   ✅ Found marker, clicking...")
            marker.click()
            
            # Wait for new tab to open
            time.sleep(1.0)
            
            # Check for new window/tab
            new_windows = set(self.driver.window_handles) - original_windows
            
            if not new_windows:
                print("   ❌ No new tab opened")
                return None
            
            # Switch to new tab
            new_window = new_windows.pop()
            self.driver.switch_to.window(new_window)
            time.sleep(0.3)
            
            # Get URL and extract coordinates
            url = self.driver.current_url
            print(f"   🔗 URL: {url[:80]}...")
            
            # Close tab and switch back FIRST
            self.driver.close()
            self.driver.switch_to.window(original_window)
            
            # Extract viewpoint coordinates
            if 'viewpoint=' in url:
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'viewpoint' in params:
                    coords_str = params['viewpoint'][0]
                    parts = coords_str.split(',')
                    if len(parts) == 2:
                        lat = float(parts[0])
                        lng = float(parts[1])
                        print(f"   ✅ Extracted: ({lat:.6f}, {lng:.6f})")
                        return (lat, lng)
            
            print("   ❌ No viewpoint in URL")
            return None
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if original_window:
                try:
                    self.driver.switch_to.window(original_window)
                except:
                    pass
            return None
    
    def _append_to_csv(self, result: RoundResult):
        """Append a result to the CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.round_num,
                result.timestamp,
                f"{result.predicted_lat:.6f}",
                f"{result.predicted_lng:.6f}",
                f"{result.true_lat:.6f}" if result.true_lat else "",
                f"{result.true_lng:.6f}" if result.true_lng else "",
                f"{result.distance_km:.2f}" if result.distance_km else "",
                result.score if result.score else "",
                self.stage1_checkpoint or "",
                self.stage2_checkpoint or ""
            ])
    
    def get_summary(self) -> dict:
        """Get summary statistics for all recorded rounds."""
        if not self.results:
            return {}
        
        distances = [r.distance_km for r in self.results if r.distance_km is not None]
        scores = [r.score for r in self.results if r.score is not None]
        
        summary = {
            "total_rounds": len(self.results),
            "rounds_with_results": len(distances),
        }
        
        if distances:
            summary["avg_distance_km"] = sum(distances) / len(distances)
            summary["min_distance_km"] = min(distances)
            summary["max_distance_km"] = max(distances)
        
        if scores:
            summary["total_score"] = sum(scores)
            summary["avg_score"] = sum(scores) / len(scores)
        
        return summary
    
    def print_summary(self):
        """Print game summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 GAME SUMMARY")
        print("="*60)
        
        if not summary:
            print("No results recorded.")
            return
        
        print(f"Rounds played: {summary['total_rounds']}")
        print(f"Rounds with results: {summary['rounds_with_results']}")
        
        if 'avg_distance_km' in summary:
            print(f"\nDistance Statistics:")
            print(f"   Average: {summary['avg_distance_km']:.1f} km")
            print(f"   Best: {summary['min_distance_km']:.1f} km")
            print(f"   Worst: {summary['max_distance_km']:.1f} km")
        
        if 'total_score' in summary:
            print(f"\nScore Statistics:")
            print(f"   Total: {summary['total_score']}")
            print(f"   Average: {summary['avg_score']:.0f}")
        
        print(f"\nResults saved to: {self.csv_path}")
        print("="*60)
    
    def close(self):
        """Clean up (don't close browser, just disconnect)."""
        if self.driver:
            self.driver = None


class SimpleResultsTracker:
    """
    Simple results tracker without Selenium.
    
    Records predictions and allows manual entry of true locations/scores.
    Also supports OCR-based score extraction from screenshots.
    """
    
    def __init__(self, output_dir: str = "results", stage1_checkpoint: str = None, stage2_checkpoint: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[RoundResult] = []

        # Checkpoint information
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint

        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"game_results_{self.session_timestamp}.csv"

        self._init_csv()
        print(f"📊 Simple results tracker initialized: {self.csv_path}")
        if self.stage1_checkpoint:
            print(f"   Stage1 checkpoint: {self.stage1_checkpoint}")
        if self.stage2_checkpoint:
            print(f"   Stage2 checkpoint: {self.stage2_checkpoint}")
    
    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp',
                'predicted_lat', 'predicted_lng',
                'true_lat', 'true_lng',
                'distance_km', 'score',
                'stage1_checkpoint', 'stage2_checkpoint'
            ])
    
    def record_round(
        self,
        round_num: int,
        pred_lat: float,
        pred_lng: float,
        true_lat: Optional[float] = None,
        true_lng: Optional[float] = None,
        score: Optional[int] = None
    ):
        """Record a round result."""
        distance_km = None
        if true_lat is not None and true_lng is not None:
            distance_km = haversine_distance(pred_lat, pred_lng, true_lat, true_lng)
        
        result = RoundResult(
            round_num=round_num,
            predicted_lat=pred_lat,
            predicted_lng=pred_lng,
            true_lat=true_lat,
            true_lng=true_lng,
            distance_km=distance_km,
            score=score
        )
        self.results.append(result)
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.round_num,
                result.timestamp,
                f"{result.predicted_lat:.6f}",
                f"{result.predicted_lng:.6f}",
                f"{result.true_lat:.6f}" if result.true_lat else "",
                f"{result.true_lng:.6f}" if result.true_lng else "",
                f"{result.distance_km:.2f}" if result.distance_km else "",
                result.score if result.score else "",
                self.stage1_checkpoint or "",
                self.stage2_checkpoint or ""
            ])
        
        return result
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.results:
            return {}
        
        distances = [r.distance_km for r in self.results if r.distance_km is not None]
        scores = [r.score for r in self.results if r.score is not None]
        
        summary = {"total_rounds": len(self.results)}
        
        if distances:
            summary["avg_distance_km"] = sum(distances) / len(distances)
            summary["min_distance_km"] = min(distances)
            summary["max_distance_km"] = max(distances)
        
        if scores:
            summary["total_score"] = sum(scores)
            summary["avg_score"] = sum(scores) / len(scores)
        
        return summary

