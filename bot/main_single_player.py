import pyautogui
import yaml
import os
import requests
import base64
from io import BytesIO
from time import sleep
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from select_regions import get_coords
from geoguessr_bot import GeoBot


class MLGeoBot(GeoBot):
    """GeoBot that uses server-side ML model instead of LLM"""

    def __init__(self, screen_regions, player=1, api_url="http://localhost:5000/api/v1/predict"):
        # Initialize without model since we use API
        self.screen_regions = screen_regions
        self.player = player
        self.api_url = api_url
        self.screen_xywh = screen_regions[f'player_{player}']['panorama']

        # Calculate map boundaries for coordinate conversion
        map_region = screen_regions[f'player_{player}']['map']
        self.map_x = map_region[0]
        self.map_y = map_region[1]
        self.map_width = map_region[2]
        self.map_height = map_region[3]

    def predict_location(self, screenshot):
        """Send screenshot to ML server and get prediction"""
        # Convert PIL to base64
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Send to server
        response = requests.post(
            self.api_url,
            json={"image": f"data:image/png;base64,{img_base64}"},
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        result = response.json()
        lat = result["results"]["lat"]
        lng = result["results"]["lng"]

        print(f"🎯 ML Model Prediction: {lat:.4f}, {lng:.4f}")
        return lat, lng

    def latlng_to_screen_coords(self, lat, lng):
        """Convert lat/lng to screen coordinates for clicking on GeoGuessr map"""
        # GeoGuessr uses a Web Mercator-like projection
        # This is a simplified conversion - may need calibration for your screen

        # Normalize longitude to 0-1 range (-180 to 180)
        lng_ratio = (lng + 180) / 360

        # For latitude, use Mercator projection approximation
        # Clamp latitude to valid Mercator range
        lat = max(-85, min(85, lat))
        lat_rad = lat * 3.14159 / 180
        # Mercator Y coordinate
        mercator_y = (1 - (1 / 3.14159) * (3.14159/2 - lat_rad)) / 2

        # Flip Y axis (screen coordinates go top-to-bottom)
        lat_ratio = 1 - mercator_y

        # Convert to screen coordinates within map region
        screen_x = self.map_x + int(lng_ratio * self.map_width)
        screen_y = self.map_y + int(lat_ratio * self.map_height)

        # Ensure coordinates are within map bounds
        screen_x = max(self.map_x, min(self.map_x + self.map_width, screen_x))
        screen_y = max(self.map_y, min(self.map_y + self.map_height, screen_y))

        print(f"🗺️  Map coordinates: lat={lat:.4f}, lng={lng:.4f} → screen=({screen_x}, {screen_y})")
        return screen_x, screen_y

    def select_map_location(self, lat, lng, plot=False):
        """Click on map at lat/lng coordinates"""
        screen_x, screen_y = self.latlng_to_screen_coords(lat, lng)

        print(f"🖱️  Clicking at screen coordinates: ({screen_x}, {screen_y})")

        if plot:
            # Optional: draw on screen for debugging
            pass

        # Move mouse and click
        pyautogui.moveTo(screen_x, screen_y, duration=0.5)
        pyautogui.click()

        sleep(0.5)  # Wait for click to register


def play_turn_llm(bot: GeoBot, plot: bool = False):
    """Play turn using LLM-based approach"""
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)
    screenshot_b64 = GeoBot.pil_to_base64(screenshot)
    message = GeoBot.create_message([screenshot_b64])

    response = bot.model.invoke([message])
    print(response.content)

    location = bot.extract_location_from_response(response)
    if location is None:
        # Second try
        response = bot.model.invoke([message])
        print(response.content)
        location = bot.extract_location_from_response(response)
    
    if location is not None:
        bot.select_map_location(*location, plot=plot)
    else:
        print("Error getting a location for second time")
        bot.select_map_location(x=1, y=1, plot=plot)

    # Going to the next round
    pyautogui.press(" ")
    sleep(2)


def play_turn_ml(bot: MLGeoBot, plot: bool = False):
    """Play turn using ML model approach for GeoGuessr"""
    print("🎮 Starting new round...")

    # Wait for panorama to load (look for some visual indicator)
    print("⏳ Waiting for panorama to load...")
    sleep(3)  # Give time for panorama to load

    # Take screenshot of panorama
    print("📸 Capturing panorama screenshot...")
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)

    try:
        # Get prediction from ML model
        lat, lng = bot.predict_location(screenshot)
        print(f"📍 Predicted location: {lat:.4f}, {lng:.4f}")

        # Convert to screen coordinates and click
        bot.select_map_location(lat, lng, plot=plot)

        # Wait for guess to register
        sleep(2)

    except Exception as e:
        print(f"❌ Error getting prediction: {e}")
        # Fallback: click in center of map
        center_x = bot.map_x + bot.map_width // 2
        center_y = bot.map_y + bot.map_height // 2
        print(f"🎯 Using fallback location: center of map ({center_x}, {center_y})")
        pyautogui.click(center_x, center_y)
        sleep(2)

    # Going to the next round
    print("⏭️  Moving to next round...")
    pyautogui.press(" ")
    sleep(3)  # Wait for next round to load


def play_turn(bot, plot: bool = False):
    """Universal play_turn function that works with both bot types"""
    if isinstance(bot, MLGeoBot):
        play_turn_ml(bot, plot)
    else:
        play_turn_llm(bot, plot)


def main_llm(turns=5, plot=False):
    """Main function using LLM-based approach"""
    if "screen_regions.yaml" not in os.listdir():
        screen_regions = get_coords(players=1)
    with open("screen_regions.yaml") as f:
        screen_regions = yaml.safe_load(f)

    bot = GeoBot(
        screen_regions=screen_regions,
        player=1,
        model=ChatOpenAI,  # ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic
        model_name="gpt-4o",   # gpt-4o, gemini-1.5-pro, claude-3-5-sonnet-20240620
    )

    print("🤖 Using LLM-based GeoGuessr Bot (GPT-4)")

    for turn in range(turns):
        print("\n----------------")
        print(f"Turn {turn+1}/{turns}")
        play_turn(bot=bot, plot=plot)


def detect_game_start():
    """Try to detect when GeoGuessr game starts by looking for panorama"""
    print("🔍 Looking for GeoGuessr game...")

    # Take a test screenshot to see if we can find game elements
    screenshot = pyautogui.screenshot()

    # You could add image recognition here to detect game elements
    # For now, we'll just assume the game is open and ready

    print("✅ Assuming game is ready (open GeoGuessr in browser first)")
    return True


def main_ml(turns=5, plot=False, api_url="http://localhost:5000/api/v1/predict"):
    """Main function using ML model approach for GeoGuessr"""
    if "screen_regions.yaml" not in os.listdir():
        print("❌ screen_regions.yaml not found!")
        print("Please run: python select_regions.py")
        return

    with open("screen_regions.yaml") as f:
        screen_regions = yaml.safe_load(f)

    bot = MLGeoBot(
        screen_regions=screen_regions,
        player=1,
        api_url=api_url
    )

    print("🤖 GeoGuessr ML Bot Started")
    print(f"📍 API Endpoint: {api_url}")
    print(f"🎯 Target rounds: {turns}")
    print("📋 Make sure GeoGuessr is open in your browser!")

    # Wait for user to start the game
    input("Press Enter when you're ready to start the game...")

    for turn in range(turns):
        print(f"\n{'='*50}")
        print(f"🎮 ROUND {turn+1}/{turns}")
        print(f"{'='*50}")

        try:
            play_turn(bot=bot, plot=plot)
            print(f"✅ Round {turn+1} completed successfully")
        except Exception as e:
            print(f"❌ Error in round {turn+1}: {e}")
            sleep(5)  # Wait before continuing

    print(f"\n🎉 Game completed! Played {turns} rounds.")


def main(turns=5, plot=False, mode="llm", api_url="http://localhost:5000/api/v1/predict"):
    """Main function with mode selection"""
    if mode == "ml":
        main_ml(turns=turns, plot=plot, api_url=api_url)
    else:
        main_llm(turns=turns, plot=plot)


if __name__ == "__main__":
    # Choose your approach:
    # main(turns=5, plot=True, mode="llm")  # Use GPT-4/Claude
    main(turns=5, plot=True, mode="ml")   # Use ML model (requires running server)