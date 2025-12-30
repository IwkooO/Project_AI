#!/usr/bin/env python3
"""
GeoGuessr API Client

Provides direct API access to submit guesses and interact with GeoGuessr games.
Requires authentication cookies from an active browser session.

Usage:
    # Basic usage with cookies from browser
    api = GeoGuessrAPI()
    api.load_cookies_from_browser()  # Uses cookies from Chrome
    
    # Submit a guess
    result = api.submit_guess(
        game_id="abc123xyz",
        lat=48.8566,
        lng=2.3522,
        round_number=1,
        game_type="classic"
    )
    print(f"Score: {result['roundScore']['amount']}")
"""

import json
import os
import re
import sqlite3
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal

import requests


GameType = Literal["classic", "duels", "team-duels", "battle-royale", "live-challenge"]


@dataclass
class GuessResult:
    """Result of a guess submission."""
    success: bool
    score: Optional[int] = None
    distance_meters: Optional[float] = None
    true_lat: Optional[float] = None
    true_lng: Optional[float] = None
    round_number: Optional[int] = None
    total_score: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GeoGuessrAPI:
    """
    Client for interacting with GeoGuessr's game API.
    
    Supports:
    - Classic games (solo/multiplayer)
    - Duels
    - Battle Royale
    
    Requires authentication via cookies from an active browser session.
    """
    
    # API endpoints
    BASE_URL = "https://www.geoguessr.com"
    GAME_SERVER_URL = "https://game-server.geoguessr.com"
    
    # Classic games use the v3 API (POST to same endpoint as GET)
    # Duels and Battle Royale use the game-server API
    # API endpoints for different game types
    API_ENDPOINTS = {
        "classic": "{base}/api/v3/games/{game_id}",  # v3 API - POST with token, lat, lng
        "duels": "{server}/api/duels/{game_id}/guess",
        "team-duels": "{server}/api/duels/{game_id}/guess",  # Team duels use same endpoint
        "battle-royale": "{server}/api/battle-royale/{game_id}/guess",
        "live-challenge": "{server}/api/live-challenge/{game_id}/guess",  # Different from duels
    }
    
    GAME_INFO_ENDPOINTS = {
        "classic": "{base}/api/v3/games/{game_id}",
        "duels": "{server}/api/duels/{game_id}",
        "team-duels": "{server}/api/duels/{game_id}",  # Team duels use same endpoint
        "battle-royale": "{server}/api/battle-royale/{game_id}",
        "live-challenge": "{server}/api/live-challenge/{game_id}",  # Uses live-challenge path
    }
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the GeoGuessr API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.session = requests.Session()
        self.timeout = timeout
        self._setup_headers()
    
    def _setup_headers(self):
        """Set up default headers for requests."""
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://www.geoguessr.com",
            "Referer": "https://www.geoguessr.com/",
            "x-client": "web",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        })
    
    def set_cookies(self, cookies: Dict[str, str]):
        """
        Set authentication cookies.
        
        Args:
            cookies: Dictionary of cookie name-value pairs
        """
        for name, value in cookies.items():
            self.session.cookies.set(name, value, domain=".geoguessr.com")
        print(f"✅ Set {len(cookies)} cookies")
    
    def load_cookies_from_file(self, filepath: str):
        """
        Load cookies from a JSON file.
        
        Args:
            filepath: Path to JSON file containing cookies
        
        File format:
        {
            "_ncfa": "...",
            "device_token": "...",
            ...
        }
        """
        with open(filepath, 'r') as f:
            cookies = json.load(f)
        self.set_cookies(cookies)
    
    def load_cookies_from_browser(self, browser: str = "chrome"):
        """
        Load cookies from a browser's cookie database.
        
        Args:
            browser: Browser name ("chrome" or "firefox")
        
        Note: Chrome must be closed, or use --enable-features=CookieEncryptionBypass
              Firefox cookies are usually unencrypted on Linux.
        """
        if browser.lower() == "chrome":
            self._load_chrome_cookies()
        elif browser.lower() == "firefox":
            self._load_firefox_cookies()
        else:
            raise ValueError(f"Unsupported browser: {browser}")
    
    def _load_chrome_cookies(self):
        """Load cookies from Chrome's cookie database."""
        # Common Chrome cookie paths on Linux
        cookie_paths = [
            Path.home() / ".config/google-chrome/Default/Cookies",
            Path.home() / ".config/google-chrome-beta/Default/Cookies",
            Path.home() / ".config/chromium/Default/Cookies",
        ]
        
        cookie_path = None
        for path in cookie_paths:
            if path.exists():
                cookie_path = path
                break
        
        if not cookie_path:
            raise FileNotFoundError(
                "Chrome cookie database not found. "
                "Try using load_cookies_from_file() with manually exported cookies."
            )
        
        # Copy the database to avoid locking issues
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(cookie_path, tmp.name)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            # Query for GeoGuessr cookies
            cursor.execute("""
                SELECT name, value, encrypted_value 
                FROM cookies 
                WHERE host_key LIKE '%geoguessr.com'
            """)
            
            cookies = {}
            for name, value, encrypted_value in cursor.fetchall():
                if value:
                    cookies[name] = value
                elif encrypted_value:
                    # Chrome encrypts cookies - try to decrypt
                    decrypted = self._decrypt_chrome_cookie(encrypted_value)
                    if decrypted:
                        cookies[name] = decrypted
            
            conn.close()
            
            if cookies:
                self.set_cookies(cookies)
                print(f"✅ Loaded {len(cookies)} cookies from Chrome")
            else:
                print("⚠️ No GeoGuessr cookies found in Chrome")
                print("   Make sure you're logged into GeoGuessr in Chrome")
                
        finally:
            os.unlink(tmp_path)
    
    def _decrypt_chrome_cookie(self, encrypted_value: bytes) -> Optional[str]:
        """
        Attempt to decrypt Chrome cookie.
        
        Chrome uses different encryption on different platforms.
        On Linux, it typically uses DPAPI or gnome-keyring.
        """
        # Check if it's v10 encryption (Linux)
        if encrypted_value[:3] == b'v10':
            try:
                import secretstorage
                from Crypto.Cipher import AES
                from Crypto.Protocol.KDF import PBKDF2
                
                # Get the encryption key from gnome-keyring
                connection = secretstorage.dbus_init()
                collection = secretstorage.get_default_collection(connection)
                
                for item in collection.get_all_items():
                    if 'Chrome' in item.get_label():
                        key = PBKDF2(item.get_secret(), b'saltysalt', dkLen=16, count=1)
                        
                        # Remove v10 prefix and decrypt
                        encrypted_data = encrypted_value[3:]
                        iv = b' ' * 16
                        cipher = AES.new(key, AES.MODE_CBC, iv)
                        decrypted = cipher.decrypt(encrypted_data)
                        
                        # Remove padding
                        padding_len = decrypted[-1]
                        return decrypted[:-padding_len].decode('utf-8')
            except ImportError:
                pass
            except Exception:
                pass
        
        return None
    
    def _load_firefox_cookies(self):
        """Load cookies from Firefox's cookie database."""
        # Find Firefox profile
        firefox_path = Path.home() / ".mozilla/firefox"
        
        if not firefox_path.exists():
            raise FileNotFoundError("Firefox profile directory not found")
        
        # Find the default profile
        profiles_ini = firefox_path / "profiles.ini"
        profile_path = None
        
        if profiles_ini.exists():
            import configparser
            config = configparser.ConfigParser()
            config.read(profiles_ini)
            
            for section in config.sections():
                if section.startswith("Profile"):
                    if config.get(section, "Default", fallback="0") == "1":
                        profile_path = firefox_path / config.get(section, "Path")
                        break
                    if not profile_path:
                        profile_path = firefox_path / config.get(section, "Path")
        
        if not profile_path:
            # Try to find any .default profile
            for path in firefox_path.iterdir():
                if path.is_dir() and ".default" in path.name:
                    profile_path = path
                    break
        
        if not profile_path:
            raise FileNotFoundError("Firefox profile not found")
        
        cookie_db = profile_path / "cookies.sqlite"
        if not cookie_db.exists():
            raise FileNotFoundError(f"Firefox cookie database not found: {cookie_db}")
        
        # Copy to avoid locking
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            shutil.copy2(cookie_db, tmp.name)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name, value 
                FROM moz_cookies 
                WHERE host LIKE '%geoguessr.com'
            """)
            
            cookies = {name: value for name, value in cursor.fetchall()}
            conn.close()
            
            if cookies:
                self.set_cookies(cookies)
                print(f"✅ Loaded {len(cookies)} cookies from Firefox")
            else:
                print("⚠️ No GeoGuessr cookies found in Firefox")
                
        finally:
            os.unlink(tmp_path)
    
    def load_cookies_from_selenium(self, driver):
        """
        Load cookies from a Selenium WebDriver.
        
        Args:
            driver: Selenium WebDriver instance connected to GeoGuessr
        """
        selenium_cookies = driver.get_cookies()
        cookies = {c['name']: c['value'] for c in selenium_cookies}
        self.set_cookies(cookies)
    
    def export_cookies(self, filepath: str):
        """
        Export current cookies to a JSON file.
        
        Args:
            filepath: Output file path
        """
        cookies = dict(self.session.cookies)
        with open(filepath, 'w') as f:
            json.dump(cookies, f, indent=2)
        print(f"✅ Exported {len(cookies)} cookies to {filepath}")
    
    def _get_game_type_from_url(self, url: str) -> GameType:
        """Determine game type from URL."""
        if "/duels/" in url:
            # Note: team-duels also uses /duels/ URL - differentiate via API response
            return "duels"
        elif "/battle-royale/" in url:
            return "battle-royale"
        elif "/live-challenge/" in url:
            return "live-challenge"
        else:
            return "classic"
    
    def _get_game_id_from_url(self, url: str) -> Optional[str]:
        """Extract game ID from URL."""
        patterns = [
            r'/game/([A-Za-z0-9]+)',
            r'/duels/([A-Za-z0-9\-]+)',  # UUID format for duels
            r'/battle-royale/([A-Za-z0-9\-]+)',
            r'/live-challenge/([A-Za-z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_game_info(self, game_id: str, game_type: GameType = "classic") -> Optional[Dict[str, Any]]:
        """
        Get information about a game.
        
        Args:
            game_id: The game ID
            game_type: Type of game ("classic", "duels", or "battle-royale")
        
        Returns:
            Game info dictionary or None if failed
        """
        if game_type == "classic":
            url = self.GAME_INFO_ENDPOINTS["classic"].format(
                base=self.BASE_URL, 
                game_id=game_id
            )
        else:
            url = self.GAME_INFO_ENDPOINTS[game_type].format(
                server=self.GAME_SERVER_URL, 
                game_id=game_id
            )
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get game info: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error getting game info: {e}")
            return None
    
    def _parse_game_response(
        self, 
        data: Dict[str, Any], 
        game_type: GameType, 
        round_number: int
    ) -> GuessResult:
        """
        Parse game response based on game type.
        
        Duels/Team-Duels have a different structure than Live Challenge:
        - Duels: guesses is a dict keyed by player ID with nested score/distance
        - Live Challenge: guesses is a list with flat score/distance
        
        Args:
            data: API response data
            game_type: Type of game
            round_number: Current round number
            
        Returns:
            GuessResult with parsed data
        """
        current_round = data.get("currentRoundNumber") or data.get("round") or round_number
        
        score = None
        distance_m = None
        true_lat = None
        true_lng = None
        total_score = None
        
        guesses = data.get("guesses", {})
        rounds = data.get("rounds", [])
        
        # Determine response format based on guesses structure
        if isinstance(guesses, dict):
            # DUELS FORMAT: guesses is dict keyed by player ID
            # Structure: guesses[playerId] = [{roundNumber, lat, lng, roundScore: {amount}, distance: {meters: {amount}}}]
            for player_id, player_guesses in guesses.items():
                if isinstance(player_guesses, list):
                    for guess in player_guesses:
                        if isinstance(guess, dict):
                            guess_round = guess.get("roundNumber") or guess.get("round")
                            if guess_round == round_number:
                                # Extract score from roundScore.amount
                                round_score = guess.get("roundScore")
                                if isinstance(round_score, dict):
                                    score = round_score.get("amount") or round_score.get("points")
                                elif round_score is not None:
                                    score = round_score
                                
                                # Extract distance from distance.meters.amount
                                distance = guess.get("distance")
                                if isinstance(distance, dict):
                                    meters = distance.get("meters")
                                    if isinstance(meters, dict):
                                        distance_m = meters.get("amount")
                                    elif meters is not None:
                                        distance_m = meters
                                elif distance is not None:
                                    distance_m = distance
                                
                                if score is not None:
                                    break
                    if score is not None:
                        break
            
            # Extract true location for duels (direct in rounds or in panorama object)
            if rounds and current_round and current_round <= len(rounds):
                round_data = rounds[current_round - 1]
                if isinstance(round_data, dict):
                    # Try direct lat/lng
                    true_lat = round_data.get("lat")
                    true_lng = round_data.get("lng")
                    
                    # Try panorama nested object
                    if true_lat is None:
                        panorama = round_data.get("panorama")
                        if isinstance(panorama, dict):
                            true_lat = panorama.get("lat")
                            true_lng = panorama.get("lng")
            
            # Get total score from teams (for team-duels) or directly
            teams = data.get("teams", [])
            if teams:
                # Team duels format - find our team's score
                for team in teams:
                    if isinstance(team, dict):
                        team_score = team.get("health") or team.get("totalScore")
                        if isinstance(team_score, dict):
                            total_score = team_score.get("amount")
                        elif team_score is not None:
                            total_score = team_score
                        break  # Just get first team for now
            else:
                total_score = data.get("totalScore")
                
        elif isinstance(guesses, list):
            # LIVE CHALLENGE FORMAT: guesses is a list with flat structure
            # Structure: [{roundNumber, lat, lng, score, distance}]
            for guess in guesses:
                if isinstance(guess, dict):
                    guess_round = guess.get("roundNumber") or guess.get("round")
                    if guess_round == round_number:
                        score = guess.get("score")
                        distance_m = guess.get("distance")
                        break
            
            # If no round match, try latest guess
            if score is None and guesses:
                latest_guess = guesses[-1]
                score = latest_guess.get("score")
                distance_m = latest_guess.get("distance")
            
            # Extract true location for live-challenge (nested in answer.coordinateAnswerPayload.coordinate)
            if rounds and current_round and current_round <= len(rounds):
                round_data = rounds[current_round - 1]
                if isinstance(round_data, dict):
                    # Try answer.coordinateAnswerPayload.coordinate
                    answer = round_data.get("answer", {})
                    coord_payload = answer.get("coordinateAnswerPayload", {})
                    coord = coord_payload.get("coordinate", {})
                    if coord:
                        true_lat = coord.get("lat")
                        true_lng = coord.get("lng")
                    
                    # Fallback: check question.panoramaQuestionPayload.panorama
                    if true_lat is None:
                        question = round_data.get("question", {})
                        pano_payload = question.get("panoramaQuestionPayload", {})
                        pano = pano_payload.get("panorama", {})
                        if pano:
                            true_lat = pano.get("lat")
                            true_lng = pano.get("lng")
            
            total_score = data.get("totalScore")
        
        return GuessResult(
            success=True,
            score=score,
            distance_meters=distance_m,
            true_lat=true_lat,
            true_lng=true_lng,
            round_number=current_round,
            total_score=total_score,
            raw_response=data
        )
    
    def submit_guess(
        self,
        game_id: str,
        lat: float,
        lng: float,
        round_number: int,
        game_type: GameType = "classic",
    ) -> GuessResult:
        """
        Submit a guess to a GeoGuessr game.
        
        Args:
            game_id: The game ID from the URL
            lat: Latitude of the guess (-90 to 90)
            lng: Longitude of the guess (-180 to 180)
            round_number: Current round number (1-indexed)
            game_type: Type of game ("classic", "duels", or "battle-royale")
        
        Returns:
            GuessResult object with score and distance info
        """
        # Build endpoint URL
        endpoint_template = self.API_ENDPOINTS.get(game_type)
        if not endpoint_template:
            return GuessResult(
                success=False,
                error=f"Unknown game type: {game_type}"
            )
        
        # Classic games use the base URL, others use game-server
        if game_type == "classic":
            url = endpoint_template.format(
                base=self.BASE_URL,
                game_id=game_id
            )
            # Classic v3 API uses different payload format
            payload = {
                "token": game_id,
                "lat": lat,
                "lng": lng,
                "timedOut": False,
            }
        else:
            # Duels, Battle Royale, and Live Challenge all use game-server
            url = endpoint_template.format(
                server=self.GAME_SERVER_URL,
                game_id=game_id
            )
            # Duels/Battle Royale/Live Challenge use roundNumber format
            payload = {
                "lat": lat,
                "lng": lng,
                "roundNumber": round_number,
            }
        
        # Update referer for this specific game
        self.session.headers["Referer"] = f"{self.BASE_URL}/game/{game_id}"
        
        print(f"📡 POST {url}")
        print(f"📍 Payload: lat={lat:.6f}, lng={lng:.6f}, round={round_number}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            print(f"📊 Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract results - v3 API returns data in player.guesses array
                if game_type == "classic" and "player" in data:
                    guesses = data.get("player", {}).get("guesses", [])
                    if guesses:
                        latest = guesses[-1]
                        round_score = latest.get("roundScore", {})
                        # Get distance from roundScoreInMeters or calculate
                        distance_m = latest.get("roundScoreInMeters")
                        
                        # Get true location from the rounds array
                        rounds = data.get("rounds", [])
                        true_lat = None
                        true_lng = None
                        if len(guesses) <= len(rounds):
                            round_data = rounds[len(guesses) - 1]
                            true_lat = round_data.get("lat")
                            true_lng = round_data.get("lng")
                        
                        result = GuessResult(
                            success=True,
                            score=round_score.get("amount"),
                            distance_meters=distance_m,
                            true_lat=true_lat,
                            true_lng=true_lng,
                            round_number=data.get("round"),
                            total_score=data.get("player", {}).get("totalScore", {}).get("amount"),
                            raw_response=data
                        )
                    else:
                        result = GuessResult(
                            success=True,
                            raw_response=data
                        )
                else:
                    # Handle different game types with their specific response structures
                    result = self._parse_game_response(data, game_type, round_number)
                
                print(f"✅ Guess accepted!")
                if result.score:
                    print(f"   Score: {result.score} points")
                if result.distance_meters:
                    print(f"   Distance: {result.distance_meters:.0f} meters")
                if result.true_lat and result.true_lng:
                    print(f"   True location: ({result.true_lat:.6f}, {result.true_lng:.6f})")
                
                return result
            
            elif response.status_code == 401:
                return GuessResult(
                    success=False,
                    error="Unauthorized - check your cookies/authentication"
                )
            
            elif response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("message", "Bad request")
                return GuessResult(
                    success=False,
                    error=f"Bad request: {error_msg}",
                    raw_response=error_data
                )
            
            else:
                return GuessResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.Timeout:
            return GuessResult(success=False, error="Request timed out")
        except requests.exceptions.RequestException as e:
            return GuessResult(success=False, error=f"Request failed: {e}")
        except json.JSONDecodeError as e:
            return GuessResult(success=False, error=f"Invalid JSON response: {e}")
    
    def submit_guess_from_url(
        self,
        url: str,
        lat: float,
        lng: float,
        round_number: int
    ) -> GuessResult:
        """
        Submit a guess using a GeoGuessr game URL.
        
        Automatically detects game type and extracts game ID.
        
        Args:
            url: Full GeoGuessr game URL
            lat: Latitude of the guess
            lng: Longitude of the guess
            round_number: Current round number
        
        Returns:
            GuessResult object
        """
        game_id = self._get_game_id_from_url(url)
        if not game_id:
            return GuessResult(
                success=False,
                error=f"Could not extract game ID from URL: {url}"
            )
        
        game_type = self._get_game_type_from_url(url)
        
        return self.submit_guess(
            game_id=game_id,
            lat=lat,
            lng=lng,
            round_number=round_number,
            game_type=game_type
        )
    
    def is_authenticated(self) -> bool:
        """
        Check if the current session is authenticated.
        
        Returns:
            True if authenticated, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/v3/profiles",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently authenticated user.
        
        Returns:
            User info dictionary or None
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/api/v3/profiles",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_duel_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of a duel game including both players' info.
        
        Args:
            game_id: The duel game ID
        
        Returns:
            Dict with duel state or None
        """
        try:
            url = f"{self.GAME_SERVER_URL}/api/duels/{game_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_battle_royale_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current state of a battle royale game.
        
        Args:
            game_id: The battle royale game ID
        
        Returns:
            Dict with game state or None
        """
        try:
            url = f"{self.GAME_SERVER_URL}/api/battle-royale/{game_id}"
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_game_state(self, game_id: str, game_type: GameType) -> Dict[str, Any]:
        """
        Get comprehensive game state info including round details.
        
        Args:
            game_id: The game ID
            game_type: Type of game
            
        Returns:
            Dict with: current_round, total_rounds, state, is_finished, 
                      game_mode (for duels: '1v1' or 'team')
        """
        game_info = self.get_game_info(game_id, game_type)
        if not game_info:
            return {
                "current_round": 1,
                "total_rounds": 0,  # Unknown
                "state": "unknown",
                "is_finished": False,
                "game_mode": "1v1",
            }
        
        # Extract common fields
        current_round = game_info.get("currentRoundNumber") or game_info.get("round") or 1
        state = game_info.get("state", "")
        is_finished = state in ["finished", "ended"]
        
        # Total rounds varies by game type
        # Try multiple field names that different game types might use
        total_rounds = 0
        
        # Try direct numberOfRounds field
        if game_info.get("numberOfRounds"):
            total_rounds = game_info.get("numberOfRounds")
        
        # Try totalRounds field
        if not total_rounds and game_info.get("totalRounds"):
            total_rounds = game_info.get("totalRounds")
        
        # Try rounds array length (but only if all rounds are present)
        rounds = game_info.get("rounds", [])
        
        # Check options for round settings
        options = game_info.get("options", {})
        if options:
            max_rounds = options.get("maxRounds") or options.get("rounds") or options.get("numberOfRounds")
            if max_rounds:
                total_rounds = max(total_rounds, max_rounds)
        
        # For live-challenge, check gameSettings
        game_settings = game_info.get("gameSettings", {})
        if game_settings:
            settings_rounds = game_settings.get("numberOfRounds") or game_settings.get("rounds")
            if settings_rounds:
                total_rounds = max(total_rounds, settings_rounds)
        
        # If we still don't have total_rounds, use rounds array length
        # but only if it seems complete (> 1) or we have no other info
        if not total_rounds and rounds:
            total_rounds = len(rounds)
        
        # For duels, check if it's team mode
        game_mode = "1v1"
        teams = game_info.get("teams", [])
        if teams:
            # Team duels have teams array
            game_mode = "team"
        
        return {
            "current_round": current_round,
            "total_rounds": total_rounds,
            "state": state,
            "is_finished": is_finished,
            "game_mode": game_mode,
            "raw": game_info,
        }
    
    def get_pano_id_for_round(
        self, 
        game_id: str, 
        round_number: int, 
        game_type: GameType = "classic"
    ) -> Optional[str]:
        """
        Get the panorama ID for a specific round.
        
        Args:
            game_id: The game ID
            round_number: Round number (1-indexed)
            game_type: Type of game
        
        Returns:
            Panorama ID string or None
        """
        game_info = self.get_game_info(game_id, game_type)
        if not game_info:
            return None
        
        try:
            rounds = game_info.get("rounds", [])
            if round_number <= len(rounds):
                round_data = rounds[round_number - 1]
                return round_data.get("panoId") or round_data.get("panoramaId")
        except (IndexError, KeyError):
            pass
        
        return None
    
    def start_new_session(self, ml_api_url: str) -> bool:
        """
        Start a new logging session on the ML server.
        
        This creates a new directory on the server for tracking results.
        
        Args:
            ml_api_url: The ML API predict endpoint URL
                       (e.g., "http://127.0.0.1:5000/api/v1/predict")
        
        Returns:
            True if session was started successfully, False otherwise
        """
        try:
            session_url = ml_api_url.replace('/predict', '/new_session')
            resp = requests.get(session_url, timeout=5)  # Try GET first
            if not resp.ok:
                resp = requests.post(session_url, timeout=5)  # Fallback to POST
            
            if resp.ok:
                data = resp.json()
                log_dir = data.get('log_dir', 'unknown')
                print(f"   📂 New session started: {log_dir}")
                return True
            else:
                print(f"   ⚠️ Failed to start new session: {resp.status_code}")
                return False
        except requests.exceptions.Timeout:
            print("   ⚠️ New session request timed out")
            return False
        except requests.exceptions.ConnectionError:
            print("   ⚠️ Cannot connect to ML server for new session")
            return False
        except Exception as e:
            print(f"   ⚠️ Error starting new session: {e}")
            return False
    
    def log_result_to_server(
        self,
        ml_api_url: str,
        result: GuessResult,
        round_number: int,
        pred_lat: float,
        pred_lng: float,
        model_name: str = "unknown",
    ) -> bool:
        """
        Log a game result to the ML server for analysis.
        
        Sends the prediction, ground truth, score, and distance to the
        ML server's logging endpoint for CSV recording and visualization.
        
        The server computes cell assignments automatically from the true location
        and model checkpoint (no client-side cell information needed).
        
        Args:
            ml_api_url: The ML API predict endpoint URL
                       (e.g., "http://127.0.0.1:5000/api/v1/predict")
            result: GuessResult from submit_guess()
            round_number: Current round number
            pred_lat: Predicted latitude
            pred_lng: Predicted longitude
            model_name: Name/identifier for the model
        
        Returns:
            True if result was logged successfully, False otherwise
        """
        if not result.success:
            print(f"   ⚠️ Cannot log failed result: {result.error}")
            return False
        
        if result.true_lat is None or result.true_lng is None:
            print("   ⚠️ Cannot log result: missing true location")
            return False
        
        try:
            log_url = ml_api_url.replace('/predict', '/log_result')
            payload = {
                "round": round_number,
                "pred_lat": pred_lat,
                "pred_lng": pred_lng,
                "true_lat": result.true_lat,
                "true_lng": result.true_lng,
                "distance_m": result.distance_meters or 0,
                "score": result.score or 0,
                "model": model_name,
            }
            
            resp = requests.post(log_url, json=payload, timeout=5)
            
            if resp.ok:
                data = resp.json()
                distance_km = data.get('logged', {}).get('distance_km', 0)
                true_cell_id = data.get('logged', {}).get('true_cell_id', 'N/A')
                print(f"   📝 Logged to server: {distance_km:.1f}km, {result.score} pts (cell: {true_cell_id})")
                return True
            else:
                print(f"   ⚠️ Failed to log result: {resp.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("   ⚠️ Log result request timed out")
            return False
        except requests.exceptions.ConnectionError:
            print("   ⚠️ Cannot connect to ML server for logging")
            return False
        except Exception as e:
            print(f"   ⚠️ Error logging result: {e}")
            return False
    
    def trigger_browser_refresh(self, driver) -> bool:
        """
        Trigger UI refresh in browser after API submission.
        
        This helps update the GeoGuessr UI to reflect API-submitted guesses.
        
        Args:
            driver: Selenium WebDriver instance
        
        Returns:
            True if refresh was triggered successfully
        """
        try:
            js_code = """
            (function() {
                // Dispatch events to trigger React re-render
                window.dispatchEvent(new Event('focus'));
                window.dispatchEvent(new Event('resize'));
                document.dispatchEvent(new Event('visibilitychange'));
                
                // Try to trigger map update
                const mapElements = document.querySelectorAll('.leaflet-container, .guess-map, [class*="guess-map"]');
                mapElements.forEach(el => {
                    el.dispatchEvent(new Event('mousemove', {bubbles: true}));
                });
                
                // Force any pending React updates
                if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
                    try {
                        window.__REACT_DEVTOOLS_GLOBAL_HOOK__.onCommitFiberRoot();
                    } catch(e) {}
                }
                
                return true;
            })();
            """
            driver.execute_script(js_code)
            return True
        except Exception as e:
            print(f"⚠️ Could not trigger browser refresh: {e}")
            return False
    
    def place_marker_via_js(self, driver, lat: float, lng: float) -> bool:
        """
        Place a guess marker on the map via JavaScript injection.
        
        This provides visual feedback in the browser when submitting via API.
        
        Args:
            driver: Selenium WebDriver instance
            lat: Latitude for the marker
            lng: Longitude for the marker
        
        Returns:
            True if marker was placed successfully
        """
        try:
            js_code = f"""
            (function() {{
                const lat = {lat};
                const lng = {lng};
                
                // Find map container
                const mapContainer = document.querySelector('.guess-map, [class*="guess-map"], .leaflet-container');
                if (!mapContainer) return false;
                
                // Try to access Leaflet map
                let map = mapContainer._leaflet_map;
                if (!map && window.L) {{
                    for (const id in (window.L._maps || {{}})) {{
                        map = window.L._maps[id];
                        break;
                    }}
                }}
                
                if (map) {{
                    // Pan to location and simulate click
                    map.setView([lat, lng], Math.max(map.getZoom(), 5));
                    
                    setTimeout(() => {{
                        const point = map.latLngToContainerPoint([lat, lng]);
                        const rect = mapContainer.getBoundingClientRect();
                        
                        const event = new MouseEvent('click', {{
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            clientX: rect.left + point.x,
                            clientY: rect.top + point.y
                        }});
                        mapContainer.dispatchEvent(event);
                    }}, 100);
                    
                    return true;
                }}
                
                return false;
            }})();
            """
            return driver.execute_script(js_code)
        except Exception as e:
            print(f"⚠️ Could not place marker: {e}")
            return False


def get_url_from_clipboard() -> str:
    """Get URL from system clipboard using xclip."""
    result = subprocess.run(
        ['xclip', '-selection', 'clipboard', '-o'],
        capture_output=True,
        text=True,
        timeout=2
    )
    return result.stdout.strip()


def interactive_demo():
    """Interactive demo showing how to use the API."""
    print("=" * 60)
    print("🌍 GeoGuessr API Client - Interactive Demo")
    print("=" * 60)
    
    api = GeoGuessrAPI()
    
    # Try to load cookies
    print("\n📍 Step 1: Load authentication cookies")
    print("   Options:")
    print("   1. Load from Chrome browser")
    print("   2. Load from Firefox browser")
    print("   3. Load from JSON file")
    print("   4. Enter manually")
    
    choice = input("\n   Enter choice (1-4): ").strip()
    
    try:
        if choice == "1":
            api.load_cookies_from_browser("chrome")
        elif choice == "2":
            api.load_cookies_from_browser("firefox")
        elif choice == "3":
            filepath = input("   Enter path to cookies.json: ").strip()
            api.load_cookies_from_file(filepath)
        elif choice == "4":
            print("   Enter cookies (one per line, format: name=value)")
            print("   Enter empty line when done:")
            cookies = {}
            while True:
                line = input("   ").strip()
                if not line:
                    break
                if "=" in line:
                    name, value = line.split("=", 1)
                    cookies[name.strip()] = value.strip()
            api.set_cookies(cookies)
    except Exception as e:
        print(f"   ⚠️ Could not load cookies: {e}")
        print("   Continuing without authentication...")
    
    # Check authentication
    print("\n📍 Step 2: Verify authentication")
    if api.is_authenticated():
        user = api.get_user_info()
        if user:
            nick = user.get("user", {}).get("nick", "Unknown")
            print(f"   ✅ Authenticated as: {nick}")
    else:
        print("   ⚠️ Not authenticated - guess submission may fail")
    
    # Get game URL
    print("\n📍 Step 3: Enter game details")
    url = input("   Enter GeoGuessr game URL (or press Enter to use clipboard): ").strip()
    
    if not url:
        try:
            url = get_url_from_clipboard()
            print(f"   Using URL from clipboard: {url}")
        except Exception:
            print("   Could not get URL from clipboard")
            return
    
    game_id = api._get_game_id_from_url(url)
    game_type = api._get_game_type_from_url(url)
    
    if not game_id:
        print("   ❌ Could not extract game ID from URL")
        return
    
    print(f"   Game ID: {game_id}")
    print(f"   Game type: {game_type}")
    
    # Get coordinates
    print("\n📍 Step 4: Enter guess coordinates")
    lat_str = input("   Latitude: ").strip()
    lng_str = input("   Longitude: ").strip()
    round_str = input("   Round number (default: 1): ").strip() or "1"
    
    try:
        lat = float(lat_str)
        lng = float(lng_str)
        round_num = int(round_str)
    except ValueError as e:
        print(f"   ❌ Invalid input: {e}")
        return
    
    # Submit guess
    print("\n📍 Step 5: Submit guess")
    result = api.submit_guess(
        game_id=game_id,
        lat=lat,
        lng=lng,
        round_number=round_num,
        game_type=game_type
    )
    
    if result.success:
        print("\n" + "=" * 40)
        print("🎉 GUESS RESULT")
        print("=" * 40)
        print(f"   Score: {result.score} points")
        print(f"   Distance: {result.distance_meters:.2f} meters")
        print(f"   True location: ({result.true_lat:.6f}, {result.true_lng:.6f})")
        if result.total_score:
            print(f"   Total score: {result.total_score}")
    else:
        print(f"\n   ❌ Guess failed: {result.error}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        interactive_demo()
    else:
        print(__doc__)
        print("\nRun with --demo for interactive mode")
        print("\nExample programmatic usage:")
        print("-" * 40)
        print("""
from geoguessr_api import GeoGuessrAPI

# Initialize API client
api = GeoGuessrAPI()

# Load cookies from JSON file
api.load_cookies_from_file("cookies.json")

# Or load from browser
# api.load_cookies_from_browser("chrome")

# Check if authenticated
if api.is_authenticated():
    print("Authenticated!")

# Submit a guess
result = api.submit_guess(
    game_id="abc123xyz",
    lat=48.8566,    # Paris
    lng=2.3522,
    round_number=1,
    game_type="classic"  # or "duels" or "battle-royale"
)

if result.success:
    print(f"Score: {result.score}")
    print(f"Distance: {result.distance_meters} meters")
    print(f"True location: ({result.true_lat}, {result.true_lng})")
else:
    print(f"Error: {result.error}")
""")
