#!/usr/bin/env python3
"""
Baseline evaluator for GeoCLIP and StreetCLIP.

Given a CSV with columns [image_path, latitude, longitude], this script:
1) Loads GeoCLIP (VicenteVivan/geo-clip) and StreetCLIP (geolocal/StreetCLIP).
2) Predicts lat/lng for each image.
3) Reports median distance error and accuracies within 1/25/200/750/2500 km.

Example:
python scripts/inference/baseline_geoclip_streetclip.py \
    --csv data/sample.csv \
    --models geoclip streetclip \
    --device cuda
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from geoclip import GeoCLIP
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.evaluation import accuracy_within_threshold, haversine_distance


THRESHOLDS_KM = [1.0, 25.0, 200.0, 750.0, 2500.0]


def load_dataframe(csv_path: Path, image_col: str, lat_col: str, lng_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing_cols = {image_col, lat_col, lng_col} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df = df[[image_col, lat_col, lng_col]].copy()
    df[image_col] = df[image_col].apply(Path)
    return df


def format_coord(lat: float, lng: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lng >= 0 else "W"
    return f"{abs(lat):.1f}°{ns}, {abs(lng):.1f}°{ew}"


def build_grid(step_deg: float) -> List[Tuple[float, float, str]]:
    coords: List[Tuple[float, float, str]] = []
    lat_vals = torch.arange(-90, 90 + 1e-3, step_deg)
    lng_vals = torch.arange(-180, 180 + 1e-3, step_deg)
    for lat in lat_vals:
        for lng in lng_vals:
            coords.append((float(lat), float(lng), format_coord(float(lat), float(lng))))
    return coords


class StreetClipGrid:
    """Simple StreetCLIP baseline using text similarity against a coarse grid."""

    def __init__(
        self,
        device: torch.device,
        grid_step: float = 5.0,
        candidate_csv: Optional[Path] = None,
        candidate_lat_col: str = "latitude",
        candidate_lng_col: str = "longitude",
        candidate_label_col: Optional[str] = None,
    ):
        self.device = device
        self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device)
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

        if candidate_csv:
            cand_df = pd.read_csv(candidate_csv)
            required = {candidate_lat_col, candidate_lng_col}
            missing = required - set(cand_df.columns)
            if missing:
                raise ValueError(f"candidate_csv missing columns: {missing}")
            labels = (
                cand_df[candidate_label_col].astype(str).tolist()
                if candidate_label_col and candidate_label_col in cand_df.columns
                else [format_coord(r[candidate_lat_col], r[candidate_lng_col]) for _, r in cand_df.iterrows()]
            )
            coords = list(zip(cand_df[candidate_lat_col], cand_df[candidate_lng_col], labels))
        else:
            coords = build_grid(grid_step)

        self.coord_tensor = torch.tensor([(c[0], c[1]) for c in coords], dtype=torch.float32, device=device)
        text_inputs = self.processor(
            text=[c[2] for c in coords],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
            self.text_feats = F.normalize(text_feats, dim=-1)

    @torch.no_grad()
    def predict_batch(self, image_paths: Sequence[Path], batch_size: int = 8) -> torch.Tensor:
        preds: List[torch.Tensor] = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="StreetCLIP", leave=False):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_feats = self.model.get_image_features(**inputs)
            image_feats = F.normalize(image_feats, dim=-1)

            logits = image_feats @ self.text_feats.T
            top_indices = logits.argmax(dim=1)
            preds.append(self.coord_tensor[top_indices].to("cpu"))
        return torch.cat(preds, dim=0)


@torch.no_grad()
def predict_geoclip(image_paths: Iterable[Path], device: torch.device) -> torch.Tensor:
    model = GeoCLIP().to(device)
    model.eval()
    preds: List[torch.Tensor] = []
    for path in tqdm(image_paths, desc="GeoCLIP", leave=False):
        pred, _ = model.predict(str(path), top_k=1)
        pred_tensor = torch.tensor(pred[0], dtype=torch.float32)
        preds.append(pred_tensor)
    return torch.stack(preds, dim=0)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    distances = haversine_distance(preds, targets)
    metrics = {"median_km": float(distances.median().item())}
    for t in THRESHOLDS_KM:
        metrics[f"acc@{int(t)}km"] = accuracy_within_threshold(distances, t)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline GeoCLIP/StreetCLIP geolocation evaluator.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with image_path, latitude, longitude columns.")
    parser.add_argument("--image_col", type=str, default="image_path", help="Column with image paths.")
    parser.add_argument("--lat_col", type=str, default="lat", help="Latitude column name.")
    parser.add_argument("--lng_col", type=str, default="lng", help="Longitude column name.")
    parser.add_argument("--models", nargs="+", default=["geoclip", "streetclip"], choices=["geoclip", "streetclip"])
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Defaults to cuda if available.")
    parser.add_argument("--grid_step", type=float, default=5.0, help="StreetCLIP grid step in degrees.")
    parser.add_argument("--candidate_csv", type=Path, default=None, help="Optional StreetCLIP candidate CSV.")
    parser.add_argument("--candidate_lat_col", type=str, default="latitude", help="Lat column for candidate CSV.")
    parser.add_argument("--candidate_lng_col", type=str, default="longitude", help="Lng column for candidate CSV.")
    parser.add_argument("--candidate_label_col", type=str, default=None, help="Optional label column for candidate CSV.")
    parser.add_argument("--streetclip_batch_size", type=int, default=8, help="Batch size for StreetCLIP feature extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_dataframe(args.csv, args.image_col, args.lat_col, args.lng_col)
    if df.empty:
        raise ValueError("CSV is empty.")

    missing = [p for p in df[args.image_col] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files do not exist. First missing: {missing[0]}")

    targets = torch.tensor(df[[args.lat_col, args.lng_col]].values, dtype=torch.float32)

    if "geoclip" in args.models:
        geoclip_preds = predict_geoclip(df[args.image_col].tolist(), device)
        geoclip_metrics = compute_metrics(geoclip_preds, targets)
        print("\nGeoCLIP Metrics")
        print("----------------")
        for k, v in geoclip_metrics.items():
            print(f"{k}: {v:.4f}")

    if "streetclip" in args.models:
        streetclip = StreetClipGrid(
            device=device,
            grid_step=args.grid_step,
            candidate_csv=args.candidate_csv,
            candidate_lat_col=args.candidate_lat_col,
            candidate_lng_col=args.candidate_lng_col,
            candidate_label_col=args.candidate_label_col,
        )
        streetclip_preds = streetclip.predict_batch(df[args.image_col].tolist(), batch_size=args.streetclip_batch_size)
        streetclip_metrics = compute_metrics(streetclip_preds, targets)
        print("\nStreetCLIP Metrics")
        print("------------------")
        for k, v in streetclip_metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
