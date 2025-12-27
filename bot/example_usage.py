#!/usr/bin/env python3
"""
Example usage of updated trackers with checkpoint logging.

This shows how to initialize the trackers with checkpoint information
and how to query the API server for checkpoint details.
"""

import requests
from pyautogui_tracker import PyAutoGUIResultsTracker
from results_tracker import ResultsTracker, SimpleResultsTracker


def get_checkpoint_info_from_api(api_url: str = "http://localhost:5000"):
    """Query the API server for checkpoint information."""
    try:
        response = requests.get(f"{api_url}/api/v1/checkpoints")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get checkpoint info: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error querying API: {e}")
        return {}


def example_usage():
    """Example of how to use the updated trackers."""

    # Method 1: Get checkpoint info from API server
    print("Querying API server for checkpoint information...")
    checkpoint_info = get_checkpoint_info_from_api()

    stage1_checkpoint = checkpoint_info.get("stage1_checkpoint")
    stage2_checkpoint = checkpoint_info.get("stage2_checkpoint")

    print(f"Stage1 checkpoint: {stage1_checkpoint}")
    print(f"Stage2 checkpoint: {stage2_checkpoint}")

    # Method 2: Manually specify checkpoint paths
    # stage1_checkpoint = "/path/to/stage1_checkpoint.pt"
    # stage2_checkpoint = "/path/to/stage2_checkpoint.pt"

    # Initialize trackers with checkpoint information
    print("\nInitializing trackers with checkpoint info...")

    # PyAutoGUI tracker
    pyautogui_tracker = PyAutoGUIResultsTracker(
        output_dir="results",
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint
    )

    # Selenium-based tracker
    selenium_tracker = ResultsTracker(
        output_dir="results",
        chrome_debug_port=9222,
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint
    )

    # Simple tracker (for manual entry)
    simple_tracker = SimpleResultsTracker(
        output_dir="results",
        stage1_checkpoint=stage1_checkpoint,
        stage2_checkpoint=stage2_checkpoint
    )

    print("\nTrackers initialized. CSV files will now include checkpoint columns:")
    print("- stage1_checkpoint")
    print("- stage2_checkpoint")

    # Example of recording a prediction
    # pyautogui_tracker.record_prediction(round_num=1, pred_lat=40.7128, pred_lng=-74.0060)

    # Example of recording a complete round (for simple tracker)
    # simple_tracker.record_round(
    #     round_num=1,
    #     pred_lat=40.7128,
    #     pred_lng=-74.0060,
    #     true_lat=40.7589,
    #     true_lng=-73.9851,
    #     score=4500
    # )

    print("\nThe CSV files will be saved with checkpoint information in each row.")


if __name__ == "__main__":
    example_usage()
