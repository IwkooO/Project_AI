#!/usr/bin/env python3
"""
GeoGuessr Bot Main Script

Usage:
    1. Make sure the API server is running (via SSH tunnel to localhost:5000)
    2. Open GeoGuessr in your browser and start a game
    3. Run: python main_bot.py
    4. If first time: follow prompts to configure screen regions
    5. The bot will automatically play the specified number of turns

Options:
    --turns N       Number of turns to play (default: 5)
    --plot          Save minimap plots for debugging
    --reconfigure   Force reconfiguration of screen regions
"""

import os
import sys
import yaml
import argparse
from time import sleep

from geoguessr_bot import GeoBot, play_turn
from select_regions import get_coords


def check_api_connection() -> bool:
    """Check if the API server is reachable."""
    import requests
    
    print("🔌 Checking API connection...")
    
    # Simple test image (1x1 red pixel)
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    try:
        response = requests.post(
            "http://127.0.0.1:5000/api/v1/predict",
            json={"image": f"data:image/png;base64,{test_image}"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API connected! Test response: lat={result['results']['lat']:.2f}, lng={result['results']['lng']:.2f}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server at http://127.0.0.1:5000")
        print("   Make sure:")
        print("   1. The API server job is running on the cluster")
        print("   2. SSH tunnel is active: ssh -L 5000:<compute-node>:5000 -N user@server")
        return False
    except Exception as e:
        print(f"❌ API connection error: {e}")
        return False


def load_or_create_config(force_reconfigure: bool = False) -> dict:
    """Load screen regions config or create it interactively."""
    config_file = "screen_regions.yaml"
    
    if os.path.exists(config_file) and not force_reconfigure:
        print(f"📄 Loading configuration from {config_file}")
        with open(config_file) as f:
            return yaml.safe_load(f)
    else:
        print("🔧 No configuration found. Starting interactive setup...")
        return get_coords(players=1)


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr Bot using ML Model")
    parser.add_argument("--turns", type=int, default=5, help="Number of turns to play")
    parser.add_argument("--plot", action="store_true", help="Save minimap plots")
    parser.add_argument("--reconfigure", action="store_true", help="Force screen region reconfiguration")
    parser.add_argument("--skip-api-check", action="store_true", help="Skip API connection check")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤖 GeoGuessr Bot - Stage 2 ML Model")
    print("="*60)
    
    # Check API connection
    if not args.skip_api_check:
        if not check_api_connection():
            print("\n⚠️ API connection failed. Fix the connection and try again.")
            print("   Or run with --skip-api-check to skip this check.")
            sys.exit(1)
    
    # Load configuration
    screen_regions = load_or_create_config(args.reconfigure)
    
    # Create bot
    bot = GeoBot(screen_regions=screen_regions, player=1)
    
    print(f"\n🎮 Starting bot for {args.turns} turns...")
    print("   Make sure the GeoGuessr game window is visible!")
    print("   Press Ctrl+C to stop at any time.")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...")
        sleep(1)
    
    # Play turns
    successful = 0
    for turn in range(1, args.turns + 1):
        try:
            if play_turn(bot, turn, plot=args.plot):
                successful += 1
        except KeyboardInterrupt:
            print("\n\n⏹️ Bot stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error on turn {turn}: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"🏁 Bot finished! {successful}/{args.turns} turns completed successfully.")
    print("="*60)


if __name__ == "__main__":
    main()

