# Query Sparse vs MIL Mixed: Unexpected Learning Speed Difference

## Research Question

We observe a **dramatic learning speed difference** between two conceptually similar models for concept prediction:

- **MIL Mixed** (`CBM_MIL_Mixed`): Reaches high validation accuracy (~0.8-0.9) within **9 epochs**
- **Query Sparse** (`CBM_QuerySparse`): Only reaches ~**0.39 validation accuracy** after 9 epochs

Both models use patch-level evidence scoring with attention mechanisms, yet MIL Mixed learns **2-3x faster**. The final accuracy gap is also ~0.07 on both top-1 and top-5 metrics.

**Question**: What architectural or algorithmic differences cause this learning speed gap? Why does hard top-k selection with logsumexp learn so much faster than attention-weighted sum over all patches?

## Model Architectures

### MIL Mixed: Hard Top-K Selection with Logsumexp

**File**: `src/models/cbm_mil_mixed.py`

**Concept Head Initialization** (lines 53-97):
```python
class ConceptHeadMILMixed(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        mil_topk: int = 8,
        mil_tau: float = 0.1,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.mil_topk = mil_topk
        self.mil_tau = mil_tau

        # Project patch tokens into compact concept space
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, concept_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Patch mixing in concept space
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
        )

        # One prototype per concept in concept space (scores patches)
        self.concept_weight = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.concept_weight)
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))
```

**Forward Pass** (lines 107-138):
```python
def forward(self, pooled: torch.Tensor, patches: torch.Tensor):
    # pooled is ignored for logits; kept for API compatibility
    if patches is None:
        raise ValueError(
            "ConceptHeadMILMixed requires patch tokens. "
            "Enable --load-patch-tokens and ensure cache contains patch tokens."
        )

    # patches: [B, P, patch_dim]
    x = self.patch_proj(patches)  # [B, P, concept_dim]
    x = self.patch_mixer(x)       # [B, P, concept_dim]

    # Evidence scores: [B, P, K] then transpose to [B, K, P]
    scores_bpK = torch.matmul(x, self.concept_weight.t()) + self.concept_bias  # [B, P, K]
    scores_bKp = scores_bpK.transpose(1, 2)  # [B, K, P]

    # Top-k MIL pooling -> logits
    k = min(self.mil_topk, scores_bKp.size(-1))
    topk_vals = scores_bKp.topk(k, dim=-1).values  # [B, K, k]
    logits = self.mil_tau * torch.logsumexp(topk_vals / self.mil_tau, dim=-1)  # [B, K]

    # Attention maps: softmax over top-k evidence only (sparse, clean maps)
    topk_idx = scores_bKp.topk(k, dim=-1).indices  # [B, K, k]
    attn_logits = torch.full_like(scores_bKp, float("-inf"))
    attn_logits.scatter_(dim=-1, index=topk_idx, src=scores_bKp.gather(dim=-1, index=topk_idx))
    attn = F.softmax(attn_logits / self.mil_tau, dim=-1)  # [B, K, P]

    # Global representation for SupCon
    pooled_global = x.mean(dim=1)  # [B, concept_dim]
    hidden = self.global_proj(pooled_global)  # [B, hidden_dim]

    return logits, hidden, attn, None
```

**Key characteristics:**
- Single `concept_weight` per concept: `K × D` parameters
- Hard top-k selection: Only uses top 8 patches (out of 576 total)
- Logsumexp pooling: `logits = tau * logsumexp(topk_vals / tau)`
- Top-k masked attention: Other patches get `-inf` (zero attention)

### Query Sparse: Attention-Weighted Sum Over All Patches

**File**: `src/cbm/models/query_sparse.py`

**Concept Head Initialization** (lines 128-207):
```python
class ConceptHeadQuerySparse(nn.Module):
    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        dropout: float = 0.3,
        attn_type: str = "sparsemax",  # "sparsemax" or "softmax"
        attn_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        use_local_scores: bool = True,
        vision_proj_init_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.attn_type = attn_type
        self.attn_tau = float(attn_tau)
        self.use_local_scores = bool(use_local_scores)
        self.concept_dim = int(concept_dim)
        # Standard dot-product attention scaling (stabilizes score magnitudes).
        self.score_scale = 1.0 / math.sqrt(max(1, self.concept_dim))

        # Patch projection into concept space (CLIP-style trainable projection).
        if vision_proj_init_weight is None:
            raise ValueError(
                "vision_proj_init_weight must be provided for this baseline "
                "(expected StreetCLIP visual_projection.weight)."
            )

        proj = nn.Linear(patch_dim, concept_dim, bias=False)
        with torch.no_grad():
            proj.weight.copy_(vision_proj_init_weight.to(dtype=proj.weight.dtype))
        self.patch_proj = proj

        # Contextualize patches (patch self-attention)
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
        )

        # Concept queries (for contextual scores): [K, D]
        self.query = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.query)

        # Local prototypes (optional): [K, D]
        self.local_weight = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.local_weight)

        # Per-concept fusion gate α_k in [0,1] (initialized near 0.5)
        self.fuse_logit = nn.Parameter(torch.zeros(num_concepts))

        # Optional per-concept bias (applied to fused scores)
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def _attn(self, score_bKp: torch.Tensor) -> torch.Tensor:
        # score_bKp: [B, K, P]
        scaled = score_bKp / self.attn_tau
        if self.attn_type == "softmax":
            return F.softmax(scaled, dim=-1)
        return sparsemax(scaled, dim=-1)
```

**Forward Pass** (lines 216-257):
```python
def forward(self, patches: torch.Tensor):
    """
    Forward pass using patch tokens only.
    
    Args:
        patches: Patch tokens [B, P, patch_dim]
        
    Returns:
        tuple: (logits, hidden, attn, None)
            - logits: [B, K] concept logits
            - hidden: [B, concept_dim] hidden representation
            - attn: [B, K, P] attention weights
            - None: placeholder for compatibility
    """
    if patches is None:
        raise ValueError("ConceptHeadQuerySparse requires patch tokens (patches).")

    # patches: [B, P, patch_dim]
    x_local = self.patch_proj(patches)   # [B, P, D]
    x_ctx = self.patch_mixer(x_local)    # [B, P, D]

    # Contextual query scores: s_ctx[b,k,p] = <q_k, x_ctx[b,p]>
    s_ctx = torch.einsum("bpd,kd->bkp", x_ctx, self.query) * self.score_scale  # [B, K, P]

    if self.use_local_scores:
        s_local = torch.einsum("bpd,kd->bkp", x_local, self.local_weight) * self.score_scale  # [B, K, P]
        alpha = torch.sigmoid(self.fuse_logit).view(1, -1, 1)  # [1, K, 1]
        scores = alpha * s_ctx + (1.0 - alpha) * s_local
    else:
        scores = s_ctx

    scores = scores + self.bias.view(1, -1, 1)

    # Attention and logits (faithful)
    attn = self._attn(scores)  # [B, K, P]
    logits = (attn * scores).sum(dim=-1)  # [B, K]

    # Hidden representation (for SupCon if enabled elsewhere): mean pooled contextual patches
    hidden = x_ctx.mean(dim=1)

    return logits, hidden, attn, None
```

**Key characteristics:**
- Dual-branch architecture: `query` (K × D) + `local_weight` (K × D) + `fuse_logit` (K) + `bias` (K)
- Attention over ALL patches: All 576 patches contribute
- Attention-weighted sum: `logits = sum(attn * scores, dim=-1)`
- Fusion gate: `scores = alpha * s_ctx + (1-alpha) * s_local`

## Patch Mixer (Shared Component)

**File**: `src/models/cbm_mil_mixed.py` lines 6-38

```python
class PatchMixer(nn.Module):
    """
    Lightweight patch mixing block (Transformer encoder).
    """
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, dim]
        return self.encoder(x)
```

Both models use the same `PatchMixer` architecture for contextualizing patches.

## Sparsemax Implementation (Query Sparse)

**File**: `src/cbm/models/query_sparse.py` lines 14-108

```python
def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).
    Like softmax, maps logits -> probabilities that sum to 1, but can produce
    exact zeros (sparse distributions).
    """
    # 1) Numerical stability: shift logits by their maximum
    z = logits - logits.max(dim=dim, keepdim=True).values

    # 2) Sort z in descending order
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)

    # 3) Prefix sums of sorted logits
    z_cumsum = z_sorted.cumsum(dim)

    # 4) Determine the support size k
    r = torch.arange(1, z_sorted.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    r = r.view(*view)
    z_sorted_shifted = z_sorted - (z_cumsum - 1.0) / r
    k = (z_sorted_shifted > 0).long().sum(dim=dim, keepdim=True).clamp(min=1)

    # 5) Compute threshold tau
    k_float = k.float()
    tau = (z_cumsum.gather(dim=dim, index=k - 1) - 1.0) / k_float

    # 6) Project: p = max(z - tau, 0)
    output = (z - tau).clamp(min=0.0)
    return output
```

## Mathematical Formulations

### MIL Mixed Logit Computation

For concept `k`, batch `b`:
```python
# Step 1: Score all patches
scores[b, k, p] = <concept_weight[k], patch[b, p]> + bias[k]  # [B, K, P]

# Step 2: Select top-k patches
topk_vals = top_k(scores[b, k, :], k=8)  # [B, K, 8]

# Step 3: Logsumexp pooling
logit[b, k] = mil_tau * logsumexp(topk_vals / mil_tau)  # [B, K]
```

**Properties:**
- Only top-8 patches contribute to logits
- Logsumexp is a smooth approximation of max pooling
- Other 568 patches are completely ignored

### Query Sparse Logit Computation

For concept `k`, batch `b`:
```python
# Step 1: Contextual scores
s_ctx[b, k, p] = <query[k], contextualized_patch[b, p]> * (1/sqrt(D))  # [B, K, P]

# Step 2: Local scores (optional)
s_local[b, k, p] = <local_weight[k], raw_patch[b, p]> * (1/sqrt(D))  # [B, K, P]

# Step 3: Fusion
alpha[k] = sigmoid(fuse_logit[k])
scores[b, k, p] = alpha[k] * s_ctx[b, k, p] + (1 - alpha[k]) * s_local[b, k, p] + bias[k]

# Step 4: Attention over ALL patches
attn[b, k, p] = sparsemax(scores[b, k, :] / attn_tau)[p]  # [B, K, P]

# Step 5: Attention-weighted sum
logit[b, k] = sum_p(attn[b, k, p] * scores[b, k, p])  # [B, K]
```

**Properties:**
- All 576 patches contribute (weighted by attention)
- Attention-weighted sum is similar to weighted mean pooling
- Even low-scoring patches get non-zero attention (with softmax) or are zeroed (with sparsemax)

## Parameter Count Comparison

### MIL Mixed
```python
# Per concept:
concept_weight: K × D parameters
concept_bias: K parameters
Total: K × (D + 1) parameters
```

### Query Sparse
```python
# Per concept:
query: K × D parameters
local_weight: K × D parameters
fuse_logit: K parameters
bias: K parameters
Total: K × (2D + 2) parameters
```

**Example** (K=186 concepts, D=256):
- MIL Mixed: 186 × 257 = **47,802 parameters**
- Query Sparse: 186 × 514 = **95,604 parameters** (~2x more)

## Training Configuration

Both models are trained with:
- Same dataset (33,852 train, 4,226 val samples)
- Same patch tokens (576 patches per image, 1024-dim)
- Same concept vocabulary (186 concepts)
- Same optimizer (AdamW, lr=1e-4, weight_decay=0.02)
- Same loss (CrossEntropyLoss with label smoothing 0.1)
- Same batch size (256)

**MIL Mixed hyperparameters:**
- `mil_topk = 8`
- `mil_tau = 0.1`
- `mix_depth = 1`
- `mix_heads = 4`

**Query Sparse hyperparameters:**
- `attn_type = "sparsemax"` (or "softmax" with warmup)
- `attn_tau = 0.25` (annealed from 0.5 to 0.2)
- `mix_depth = 1`
- `mix_heads = 4`
- `use_local_scores = True`

## Observed Performance

### Learning Curves (9 epochs)

**MIL Mixed:**
- Epoch 1: Val Acc@1 ≈ 0.2
- Epoch 9: Val Acc@1 ≈ **0.8-0.9**

**Query Sparse:**
- Epoch 1: Val Acc@1 ≈ 0.0
- Epoch 9: Val Acc@1 ≈ **0.39**

### Final Performance (after full training)

**MIL Mixed:**
- Final Val Acc@1: ~0.85-0.90
- Final Val Acc@5: ~0.95-0.98

**Query Sparse:**
- Final Val Acc@1: ~0.78-0.83 (~0.07 lower)
- Final Val Acc@5: ~0.88-0.91 (~0.07 lower)

## Key Differences Summary

| Aspect | MIL Mixed | Query Sparse |
|--------|-----------|--------------|
| **Selection** | Hard top-k (8 patches) | Soft attention (all 576 patches) |
| **Pooling** | Logsumexp over top-k | Attention-weighted sum over all |
| **Parameters** | K × (D + 1) | K × (2D + 2) |
| **Architecture** | Single branch | Dual branch (query + local) |
| **Attention** | Top-k masked softmax | Full sparsemax/softmax |
| **Learning Speed** | ~0.8-0.9 val acc in 9 epochs | ~0.39 val acc in 9 epochs |

## Research Questions

1. **Why does hard top-k selection learn so much faster than attention-weighted sum?**
   - Is it the selectivity (ignoring noise) or the pooling function (logsumexp vs weighted mean)?

2. **Does the dual-branch architecture (query + local_weight + fusion) slow down learning?**
   - Or is it primarily the pooling strategy?

3. **Why does logsumexp over top-k converge faster than attention-weighted sum over all patches?**
   - Is it the smooth max property, or just the noise filtering?

4. **Would Query Sparse learn faster if we used top-k selection for logits while keeping full attention for visualization?**
   - Or is there something fundamental about the attention-weighted sum that makes learning harder?

5. **Is the parameter count difference (2x more parameters) a significant factor, or is it primarily the aggregation strategy?**

Any insights into these architectural differences and their impact on learning dynamics would be greatly appreciated!








