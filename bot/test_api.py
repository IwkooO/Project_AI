#!/usr/bin/env python3
"""
Test script to verify API connection to the ML server
"""

import requests
import base64
from PIL import Image
import io

def test_api_connection(api_url="http://localhost:5000/api/v1/predict"):
    """Test the API connection with a dummy image"""

    print(f"🧪 Testing API connection to: {api_url}")

    # Create a small test image
    test_image = Image.new('RGB', (64, 64), color='blue')
    buffered = io.BytesIO()
    test_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    try:
        response = requests.post(
            api_url,
            json={"image": f"data:image/png;base64,{img_base64}"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            lat = result["results"]["lat"]
            lng = result["results"]["lng"]
            print(f"✅ API working! Prediction: {lat:.4f}, {lng:.4f}")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure SSH tunnel is running: ssh -L 5000:gcn95:5000 -N pnair@snellius.surf.nl")
        return False

if __name__ == "__main__":
    test_api_connection()
