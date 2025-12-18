## Project context

Workspace: `/home/igodzwon/Project_AI`

This project trains a **concept prediction model** for StreetView panoramas using **StreetCLIP** embeddings and **patch-based concept heads** that output **attention maps**.

### Goal
- **High main-concept accuracy** (top-1/top-5) on the `generalized` concept label.
- **Attention overlays that make sense** spatially (faithful patch evidence).

---

## End-to-end pipeline

### 0) Map labels to generalized concepts
- Script: `scripts/data_processing/map_generalized_concepts.py`
- Input: split CSVs in `data/.../splits/` (train/val/test)
- Output: rewrites the split CSVs with a `generalized` column based on `data/concept_refinement_mapped.csv`.

### 1) Build concept vocabulary + geo priors
- Script: `scripts/data_processing/precompute_geo_priors.py`
- Output directory: `data/<concept_data_dir>/`
  - `concept_vocab.json`
  - `s2_cells.json`
  - `concept_priors.pt`

Notes:
- This repo currently trains **Phase 1 concept head only** (no geo head training).

### 2) Precompute cached StreetCLIP embeddings (pooled + patch tokens)
- Script: `scripts/data_processing/precompute_streetclip_embeddings.py`
- Model: `geolocal/StreetCLIP` (HuggingFace)
- Output directory: `.../cached_streetclip_*`
  - `{split}_pooled_embeddings.pt`
  - `{split}_patch_tokens.pt` (if `--save-patch-tokens`)
  - `{split}_metadata.json`

Typical configuration seen in logs:
- Image size: **336×336**
- Patch size: **14**
- Patch grid: **24×24 = 576** patch tokens

Performance note:
- Saving patch tokens is slow/heavy (large tensors + extra forward to fetch `last_hidden_state`).

### 3) Train concept head (Phase 1)
- Script: `scripts/training/train_cbm.py`
- Dataset: `src/data/dataset_concept.py` loads cached embeddings/tokens.
- Outputs: `checkpoints/<run>/phase1/`
  - `best_phase1.pt`
  - periodic attention visualizations under `phase1/visualizations/`

---

## Key model variants

### A) `mil_mixed` (original MIL head)
Defined in `src/models/cbm_mil_mixed.py`:
- Patch projection → patch mixer (Transformer encoder) → per-(concept,patch) evidence scores
- Logits: **top-k logsumexp** MIL pooling
- Attention: **top-k-masked softmax** over evidence (faithful)

### B) `mil_query_sparse` (query scoring + sparse attention + local/context fusion)
Added in `src/models/cbm_mil_mixed.py` as `CBM_QuerySparse` / `ConceptHeadQuerySparse`:
- Patch projection → patch mixer → per-(concept,patch) scores via learned concept vectors (“queries”)
- Optional **local score branch** (unmixed patches)
- **Fusion** of contextual + local scores with learned per-concept gate
- Attention: `softmax` or `sparsemax` over patches
- Logit: attention-weighted sum of scores (faithful)

New CLI args in `scripts/training/train_cbm.py`:
- `--model mil_query_sparse`
- `--attn-type {softmax,sparsemax}`
- `--attn-tau <float>`
- `--no-local-scores`

Stability improvements added:
- Dot-product score scaling by `1/sqrt(concept_dim)`.

### C) Attention scheduling for `mil_query_sparse`
Added in `scripts/training/train_cbm.py`:
- `--attn-warmup-epochs N`: forces **softmax** for first N epochs, then switches to `--attn-type`.
- `--anneal-attn-tau --attn-tau-start X --attn-tau-end Y`: linearly anneals `attn_tau` across epochs.

---

## Dataset / cache integrity

### Strict cache↔CSV mapping
`src/data/dataset_concept.py` was hardened so training/attention isn’t silently corrupted:
- Requires `{split}_metadata.json` + `pano_id` mapping by default.
- Refuses unsafe `cache_idx = idx` fallback unless explicitly enabled.
- Checks N consistency between `pooled_embeddings`, `patch_tokens`, and `metadata['pano_ids']`.

### Robust patch token slicing in precompute
`precompute_streetclip_embeddings.py` now:
- Computes `expected_num_patches = (image_size/patch_size)^2`.
- Strips special tokens based on `seq_len - expected_num_patches`.
- Stores `patch_size` and `expected_num_patches` in `{split}_metadata.json`.

---

## Logging changes

`scripts/training/train_cbm.py` now logs comparable metrics:
- Always prints **Train CE** and **Val CE**.
- If `rank_pu` is used, also logs rankPU stats on train/val.

---

## Concept filtering experiment

- Script: `scripts/data_processing/filter_concepts_by_min_count.py`
- Filters out concepts with fewer than `min-count` examples in **train** and applies the same whitelist to **val/test**.

---

## Job files added (typical)

- `jobs/train_cbm_mil_mixed_ce.job`: CE baseline, new cache dir and new checkpoint dir.
- `jobs/train_cbm_mil_query_sparse_ce.job`: trains `mil_query_sparse` on existing cache.
- `jobs/train_cbm_mil_query_sparse_ce_min5.job`: filter concepts below threshold + train.
- `jobs/train_cbm_mil_query_sparse_ce_sched.job`: scheduled attention (softmax warmup + tau anneal).

---

## Security note
Some job files were manually edited to include a `WANDB_API_KEY`. Do **not** commit secrets to git.
