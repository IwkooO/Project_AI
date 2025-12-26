#!/usr/bin/env python3
"""
Stage-2 probe: do Phase-1 concepts add complementary information beyond pooled embeddings?

We train lightweight geocell classifiers on:
  (A) pooled_emb  -> cell_label
  (B) [pooled_emb ; phase1_logits] -> cell_label

If (B) does not improve over (A) on val/test, then Stage-2 "both" is unlikely to beat
image-only without improving the concept signal (e.g., better Phase-1, joint training, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repo root is on sys.path when running as a script (python scripts/...).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbm.phase2.data import Stage2Dataset, collate_fn_stage2
from cbm.phase2.geocells import assign_geocells, compute_offsets, fit_semantic_geocells
from cbm.phase2.train import load_phase1_checkpoint


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProbeNet(nn.Module):
    def __init__(self, in_dim: int, num_cells: int, kind: Literal["linear", "mlp"] = "linear", dropout: float = 0.1):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(in_dim, num_cells)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_cells),
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _precompute_phase1_logits(
    phase1_model,
    loader: DataLoader,
    device: torch.device,
    num_concepts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pooled_embs: [N, pooled_dim]
      logits:      [N, num_concepts]
    """
    phase1_model.eval()
    pooled_list: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Precompute Phase1 logits"):
        pooled = batch["pooled_emb"].to(device)
        patch = batch["patch_tokens"]
        if patch is None:
            raise ValueError("patch_tokens missing; cannot compute phase1 logits for probe.")
        patch = patch.to(device)
        logits, _, _, _ = phase1_model(patch)  # [B, K]
        if logits.shape[1] != num_concepts:
            raise ValueError(f"Phase1 logits dim mismatch: got {logits.shape[1]}, expected {num_concepts}")
        pooled_list.append(pooled.detach().cpu().numpy())
        logits_list.append(logits.detach().cpu().numpy())

    pooled_all = np.concatenate(pooled_list, axis=0)
    logits_all = np.concatenate(logits_list, axis=0)
    return pooled_all, logits_all


def _collect_labels(loader: DataLoader) -> np.ndarray:
    ys: list[np.ndarray] = []
    for batch in loader:
        ys.append(batch["cell_labels"].cpu().numpy())
    return np.concatenate(ys, axis=0)


def _train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    kind: Literal["linear", "mlp"],
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    early_stop_patience: int = 3,
) -> tuple[ProbeNet, dict]:
    _set_seed(seed)

    in_dim = int(X_train.shape[1])
    num_cells = int(np.max(y_train) + 1)
    model = ProbeNet(in_dim=in_dim, num_cells=num_cells, kind=kind).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def run_epoch(X: np.ndarray, y: np.ndarray, train: bool) -> tuple[float, float]:
        model.train(train)
        order = np.arange(len(X))
        if train:
            np.random.shuffle(order)
        total_loss = 0.0
        correct = 0
        total = 0
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(X[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y[idx]).to(device=device, dtype=torch.long)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total_loss += float(loss.item()) * len(idx)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(len(idx))
        return total_loss / max(1, total), correct / max(1, total)

    best_val_acc = -1.0
    best_state = None
    bad = 0
    history = []

    for ep in range(epochs):
        tr_loss, tr_acc = run_epoch(X_train, y_train, train=True)
        va_loss, va_acc = run_epoch(X_val, y_val, train=False)
        history.append(
            {"epoch": ep + 1, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc}
        )
        if va_acc > best_val_acc + 1e-6:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_acc": best_val_acc, "history": history}


@torch.no_grad()
def _eval_probe(model: ProbeNet, X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, float]:
    model.eval()
    xb = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    yb = torch.from_numpy(y).to(device=device, dtype=torch.long)
    logits = model(xb)
    loss = float(F.cross_entropy(logits, yb).item())
    acc = float((logits.argmax(dim=1) == yb).float().mean().item())
    return loss, acc


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2 probe: pooled vs pooled+phase1_logits")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--cached-dir", required=True)
    p.add_argument("--concept-data-dir", required=True, help="Contains concept_vocab.json (for num_concepts).")
    p.add_argument("--phase1-checkpoint", required=True)

    p.add_argument("--num-cells", type=int, default=1000)
    p.add_argument("--per-country-cells", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--probe-kind", choices=["linear", "mlp"], default="linear")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-phase1-logits", action="store_true", help="Cache logits arrays to output-dir for reuse.")
    args = p.parse_args()

    _set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = Stage2Dataset(args.train_csv, args.cached_dir, split="train")
    val_ds = Stage2Dataset(args.val_csv, args.cached_dir, split="val")
    test_ds = Stage2Dataset(args.test_csv, args.cached_dir, split="test")

    # Fit/assign geocells using train split only (same as Stage2)
    centers_xyz, meta = fit_semantic_geocells(
        train_ds.coords,
        train_ds.countries,
        num_cells=int(args.num_cells),
        per_country=bool(args.per_country_cells),
        random_state=42,
    )
    train_labels = assign_geocells(train_ds.coords, train_ds.countries, centers_xyz, meta)
    val_labels = assign_geocells(val_ds.coords, val_ds.countries, centers_xyz, meta)
    test_labels = assign_geocells(test_ds.coords, test_ds.countries, centers_xyz, meta)

    train_offsets = compute_offsets(train_ds.coords, train_labels, centers_xyz)
    val_offsets = compute_offsets(val_ds.coords, val_labels, centers_xyz)
    test_offsets = compute_offsets(test_ds.coords, test_labels, centers_xyz)

    train_ds.set_labels(train_labels, train_offsets)
    val_ds.set_labels(val_labels, val_offsets)
    test_ds.set_labels(test_labels, test_offsets)

    # Dataloaders (only used to batch phase1 inference + labels)
    dl_kwargs = dict(batch_size=256, shuffle=False, num_workers=4, collate_fn=collate_fn_stage2)
    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader = DataLoader(val_ds, **dl_kwargs)
    test_loader = DataLoader(test_ds, **dl_kwargs)

    # Read num_concepts from concept vocab
    concept_vocab_path = Path(args.concept_data_dir) / "concept_vocab.json"
    with open(concept_vocab_path, "r") as f:
        concept_vocab = json.load(f)
    # Our concept_vocab.json is typically a wrapper dict containing "concept_to_idx".
    if isinstance(concept_vocab, dict) and "concept_to_idx" in concept_vocab:
        num_concepts = int(len(concept_vocab["concept_to_idx"]))
    else:
        # Fallback for older formats (e.g., plain list)
        num_concepts = int(len(concept_vocab))

    # Detect dims
    pooled_dim = int(train_ds.pooled_embeddings.shape[1])
    if train_ds.patch_tokens is None:
        raise ValueError("Cached patch tokens missing; cannot compute phase1 logits.")
    patch_dim = int(train_ds.patch_tokens.shape[2])

    device = torch.device(args.device)

    # Load frozen phase1 to get logits
    phase1_mix_lk = 5  # overwritten by typical checkpoints via Stage2 job args; for probe we don't need mix params.
    # NOTE: We rely on the checkpoint being loadable with correct config; pass minimal defaults that match your jobs.
    phase1 = load_phase1_checkpoint(
        args.phase1_checkpoint,
        device,
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=256,
        dropout=0.3,
        mil_topk=6,
        mil_tau=0.25,
        mix_depth=1,
        mix_heads=4,
        mix_mlp_ratio=2.0,
        mix_dropout=None,
        mix_local_kernel_size=phase1_mix_lk,
        proj_type="simple",
    )

    # Precompute arrays
    pooled_train, logits_train = _precompute_phase1_logits(phase1, train_loader, device, num_concepts)
    pooled_val, logits_val = _precompute_phase1_logits(phase1, val_loader, device, num_concepts)
    pooled_test, logits_test = _precompute_phase1_logits(phase1, test_loader, device, num_concepts)

    y_train = _collect_labels(train_loader)
    y_val = _collect_labels(val_loader)
    y_test = _collect_labels(test_loader)

    if args.cache_phase1_logits:
        np.save(out_dir / "pooled_train.npy", pooled_train)
        np.save(out_dir / "pooled_val.npy", pooled_val)
        np.save(out_dir / "pooled_test.npy", pooled_test)
        np.save(out_dir / "phase1_logits_train.npy", logits_train)
        np.save(out_dir / "phase1_logits_val.npy", logits_val)
        np.save(out_dir / "phase1_logits_test.npy", logits_test)
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "y_val.npy", y_val)
        np.save(out_dir / "y_test.npy", y_test)

    # Probe A: pooled only
    model_a, info_a = _train_probe(
        pooled_train,
        y_train,
        pooled_val,
        y_val,
        kind=args.probe_kind,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.probe_batch_size),
        device=device,
        seed=int(args.seed),
    )
    a_val_loss, a_val_acc = _eval_probe(model_a, pooled_val, y_val, device)
    a_test_loss, a_test_acc = _eval_probe(model_a, pooled_test, y_test, device)

    # Probe B: pooled + phase1 logits
    X_train_b = np.concatenate([pooled_train, logits_train], axis=1)
    X_val_b = np.concatenate([pooled_val, logits_val], axis=1)
    X_test_b = np.concatenate([pooled_test, logits_test], axis=1)

    model_b, info_b = _train_probe(
        X_train_b,
        y_train,
        X_val_b,
        y_val,
        kind=args.probe_kind,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.probe_batch_size),
        device=device,
        seed=int(args.seed) + 1,
    )
    b_val_loss, b_val_acc = _eval_probe(model_b, X_val_b, y_val, device)
    b_test_loss, b_test_acc = _eval_probe(model_b, X_test_b, y_test, device)

    results = {
        "settings": {
            "num_cells": int(args.num_cells),
            "per_country_cells": bool(args.per_country_cells),
            "seed": int(args.seed),
            "probe_kind": args.probe_kind,
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "pooled_dim": pooled_dim,
            "num_concepts": int(num_concepts),
            "patch_dim": patch_dim,
        },
        "probe_pooled_only": {
            "val_loss": a_val_loss,
            "val_acc": a_val_acc,
            "test_loss": a_test_loss,
            "test_acc": a_test_acc,
            "best_val_acc_during_train": float(info_a["best_val_acc"]),
        },
        "probe_pooled_plus_phase1_logits": {
            "val_loss": b_val_loss,
            "val_acc": b_val_acc,
            "test_loss": b_test_loss,
            "test_acc": b_test_acc,
            "best_val_acc_during_train": float(info_b["best_val_acc"]),
        },
        "deltas": {
            "val_acc_delta": float(b_val_acc - a_val_acc),
            "test_acc_delta": float(b_test_acc - a_test_acc),
        },
    }

    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Stage2 Probe Results ===")
    print(f"Pooled only:         val_acc={a_val_acc:.4f} test_acc={a_test_acc:.4f}")
    print(f"Pooled + logits:     val_acc={b_val_acc:.4f} test_acc={b_test_acc:.4f}")
    print(f"Delta (B-A):         val={b_val_acc - a_val_acc:+.4f} test={b_test_acc - a_test_acc:+.4f}")
    print(f"Saved: {out_dir / 'probe_results.json'}")


if __name__ == "__main__":
    main()


