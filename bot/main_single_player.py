#!/usr/bin/env python3
"""
GeoGuessr ML Bot - Main Script

This bot:
1. Takes a screenshot of the GeoGuessr panorama
2. Sends it to your ML API server (running on Snellius)
3. Gets predicted lat/lng coordinates
4. Clicks on the minimap at the predicted location
5. Confirms the guess and moves to the next round

Prerequisites:
1. Run select_regions.py first to calibrate screen positions
2. Start the ML API server on Snellius (sbatch jobs/bot/api_server.job)
3. Set up SSH tunnel: ssh -L 5000:<node>:5000 pnair@snellius.surf.nl
4. Open GeoGuessr in your browser and start a game

Usage:
    python main_single_player.py [--rounds N] [--api-url URL]
"""

import argparse
import os
import sys
import yaml
import requests

from geoguessr_bot import GeoBot, play_game
from select_regions import get_coords


def test_api_connection(api_url: str) -> bool:
    """Test if the ML API server is reachable."""
    print(f"🔌 Testing API connection to {api_url}...")
    
    # Try health endpoint first
    health_url = api_url.replace('/predict', '/health')
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.ok:
            data = resp.json()
            print(f"   ✅ API is healthy: {data}")
            return True
    except:
        pass
    
    # Try a minimal predict request
    try:
        # Send a tiny 1x1 pixel test image
        test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        resp = requests.post(
            api_url,
            json={"image": test_image},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if resp.ok:
            print(f"   ✅ API is responding (got prediction)")
            return True
        else:
            print(f"   ❌ API returned error: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API server")
        print("   Make sure:")
        print("      1. API server is running on Snellius (sbatch jobs/bot/api_server.job)")
        print("      2. SSH tunnel is active (ssh -L 5000:<node>:5000 pnair@snellius.surf.nl)")
        return False
    except requests.exceptions.Timeout:
        print("   ❌ API request timed out")
        return False
    except Exception as e:
        print(f"   ❌ API error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr ML Bot")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to play")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:5000/api/v1/predict",
                       help="ML API endpoint URL")
    parser.add_argument("--calibrate", action="store_true", help="Run screen calibration")
    parser.add_argument("--no-screenshots", action="store_true", help="Don't save screenshots")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 GEOGUESSR ML BOT")
    print("="*60)
    
    # Check if we need to calibrate
    config_file = "screen_regions.yaml"
    if args.calibrate or not os.path.exists(config_file):
        print("\n📐 Screen calibration required!")
        print("   Please have GeoGuessr open with a game started.")
        get_coords(players=1)
    
    # Load screen regions
    print(f"\n📂 Loading config from {config_file}...")
    with open(config_file) as f:
        # Use full_load to handle Python tuples from older configs
        screen_regions = yaml.full_load(f)
    
    # Convert any tuples to lists for consistency
    for key, value in screen_regions.items():
        if isinstance(value, tuple):
            screen_regions[key] = list(value)
    
    print("   ✅ Config loaded")
    
    # Test API connection
    if not test_api_connection(args.api_url):
        print("\n❌ Cannot continue without API connection.")
        print("   Please fix the connection and try again.")
        sys.exit(1)
    
    # Create bot
    bot = GeoBot(
        screen_regions=screen_regions,
        player=1,
        api_url=args.api_url
    )
    
    # Instructions
    print("\n" + "-"*60)
    print("📋 INSTRUCTIONS")
    print("-"*60)
    print("1. Make sure GeoGuessr is open in your browser")
    print("2. Start a Classic game (any map)")
    print("3. Wait for the first panorama to load")
    print("4. Press ENTER here to start the bot")
    print("-"*60)
    
    input("\n🎮 Press ENTER when ready to start...")
    
    # Play the game
    play_game(
        bot=bot,
        num_rounds=args.rounds,
        save_screenshots=not args.no_screenshots
    )


if __name__ == "__main__":
    main()
