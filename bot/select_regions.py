"""
Interactive tool to select screen regions for the GeoGuessr bot.

Run this script and follow the prompts to click on various screen locations.
The coordinates will be saved to screen_regions.yaml.
"""

import yaml
import pyautogui
from pynput import mouse


def get_click_position(prompt: str) -> list:
    """Wait for user to click and return the position as [x, y] list."""
    print(f"\n👆 {prompt}")
    print("   Click on the location...")
    
    position = [None]
    
    def on_click(x, y, button, pressed):
        if pressed:
            position[0] = [x, y]  # Use list instead of tuple for YAML compatibility
            return False  # Stop listener
    
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    print(f"   ✅ Recorded: {position[0]}")
    return position[0]


def get_coords(players: int = 1) -> dict:
    """
    Interactive coordinate collection for GeoGuessr bot.
    
    Args:
        players: Number of players (1 for solo, 2 for duels)
    
    Returns:
        Dictionary with all screen regions
    """
    print("\n" + "="*60)
    print("🎮 GeoGuessr Bot - Screen Region Configuration")
    print("="*60)
    print("\nThis tool will help you configure the screen regions.")
    print("Please have GeoGuessr open in your browser with a game started.")
    print("\nWhen prompted, click on the specified location.")
    input("\nPress Enter when ready...")
    
    regions = {}
    
    # Screen capture region
    print("\n📺 SCREEN CAPTURE REGION")
    print("-" * 40)
    regions["screen_top_left"] = get_click_position("Click TOP-LEFT corner of the game view (panorama area)")
    regions["screen_bot_right"] = get_click_position("Click BOTTOM-RIGHT corner of the game view")
    
    for player in range(1, players + 1):
        player_str = f" (Player {player})" if players > 1 else ""
        
        # Minimap region - IMPORTANT: calibrate the EXPANDED map
        print(f"\n🗺️ MINIMAP REGION{player_str}")
        print("-" * 40)
        print("   ⚠️  IMPORTANT: First HOVER over the minimap to EXPAND it!")
        print("   Then click on the corners of the EXPANDED map.")
        input("   Press Enter when map is expanded...")
        regions[f"map_top_left_{player}"] = get_click_position(f"Click TOP-LEFT corner of the EXPANDED minimap{player_str}")
        regions[f"map_bot_right_{player}"] = get_click_position(f"Click BOTTOM-RIGHT corner of the EXPANDED minimap{player_str}")
        
        # Confirm button
        print(f"\n✅ CONFIRM BUTTON{player_str}")
        print("-" * 40)
        regions[f"confirm_button_{player}"] = get_click_position(f"Click the CONFIRM/GUESS button{player_str}")
    
    # Next round button (for player 1 only in single player)
    if players == 1:
        print("\n⏭️ NEXT ROUND BUTTON")
        print("-" * 40)
        print("   After guessing, there's sometimes a 'Next Round' or 'Play Next' button.")
        print("   If your game auto-advances with SPACE key, you can skip this.")
        skip = input("   Skip next round button? (y/n): ").lower().strip()
        if skip != 'y':
            regions["next_round_button"] = get_click_position("Click the NEXT ROUND button (after making a guess)")
        else:
            regions["next_round_button"] = None
    
    # Save to file
    print("\n💾 Saving configuration...")
    with open("screen_regions.yaml", "w") as f:
        yaml.dump(regions, f, default_flow_style=False)
    
    print("✅ Configuration saved to screen_regions.yaml")
    print("\nYou can now run the bot with: python main_bot.py")
    
    return regions


if __name__ == "__main__":
    import sys
    players = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    get_coords(players=players)

