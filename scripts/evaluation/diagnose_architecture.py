import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from torch.utils.data import DataLoader

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase1.data import ConceptDataset
from cbm.joint.train import JointDataset, collate_fn_joint, build_concept_vectors
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.phase2.geocells import assign_geocells, compute_offsets

class DiagnosticModel(Stage2CrossAttentionGeoHead):
    def forward_variant(self, concept_emb, pooled_emb, cat_order="img_first", sum_order="concept_gate", gate_type="sigmoid"):
        img_h = self.pooled_proj(pooled_emb)
        concept_h = self.concept_proj(concept_emb)
        concept_h = self.concept_refinement(concept_h)
        
        if cat_order == "img_first":
            combined = torch.cat([img_h, concept_h], dim=-1)
        else:
            combined = torch.cat([concept_h, img_h], dim=-1)
            
        gate_logits = self.fusion_gate(combined)
        if gate_type == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        else:
            gate = F.softmax(gate_logits, dim=-1)
            gate = gate[:, 1:2] if gate.shape[1] == 2 else gate
        
        if sum_order == "concept_gate":
            fused_h = (1 - gate) * img_h + gate * concept_h
        else:
            fused_h = gate * img_h + (1 - gate) * concept_h
            
        cell_logits = self.cell_head(fused_h)
        return cell_logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("checkpoints/stage3_joint/stage3_joint_both_scratchgeo_dim256_pos_both_gate_ct2.5_gemini")
    checkpoint_path = checkpoint_dir / "best_joint.pt"
    phase1_run_dir = Path("checkpoints/phase1_adaptive_tau/adaptive_tau_d1_h4_dim256_pos/run_17929069")
    test_csv = phase1_run_dir / "splits" / "dataset_test.csv"
    cached_dir = "/scratch-shared/igodzwon/Project_AI/data/6921d7831744c5356b098bf7_balanced/cached_streetclip_v2"
    
    # Load metadata
    with open(phase1_run_dir / "concept_data_v2" / "concept_vocab.json", 'r') as f:
        concept_vocab = json.load(f)
    concept_names = [v for k, v in sorted({int(k): v for k, v in concept_vocab['idx_to_concept'].items()}.items())]
    
    with open(checkpoint_dir / "phase2" / "geocells.json", 'r') as f:
        geocell_data = json.load(f)
    centers_xyz = np.array(geocell_data['centers_xyz'])

    # Init models
    phase1_model = Phase1CBMTopKMil(num_concepts=186, concept_dim=256, mil_topk=6, use_per_concept_tau=True, mix_mlp_ratio=2.0).to(device)
    stage2_model = DiagnosticModel(concept_dim=256, patch_dim=1024, num_cells=1000, hidden_dim=512, mode="both", pooled_dim=768).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    phase1_model.load_state_dict(checkpoint['phase1_model_state_dict'])
    stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])
    
    concept_vectors = build_concept_vectors(phase1_model, concept_names, device, "", "").to(device)
    concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=2.5).to(device)
    if 'concept_adapter_state_dict' in checkpoint:
        concept_adapter.load_state_dict(checkpoint['concept_adapter_state_dict'])

    # Load one batch
    ds = ConceptDataset(str(test_csv), cached_dir, str(phase1_run_dir / "concept_data_v2" / "concept_vocab.json"), 
                        str(phase1_run_dir / "concept_data_v2" / "s2_cells.json"), split="test", load_pooled_embeddings=True)
    labels = assign_geocells(ds._coords, centers_xyz=centers_xyz)
    jds = JointDataset(ds, labels, compute_offsets(ds._coords, labels, centers_xyz))
    loader = DataLoader(jds, batch_size=256, shuffle=False, collate_fn=lambda b: collate_fn_joint(b, ds.pooled_embeddings))
    
    batch = next(iter(loader))
    patches, _, _, cell_labels, _, _, pooled_emb, _ = [b.to(device) if torch.is_tensor(b) else b for b in batch]
    
    with torch.no_grad():
        c_logits, _, _, _ = phase1_model(patches)
        concept_emb = concept_adapter(c_logits)
        
        # First, let's see what the actual gate outputs look like
        img_h = stage2_model.pooled_proj(pooled_emb)
        concept_h = stage2_model.concept_proj(concept_emb)
        concept_h = stage2_model.concept_refinement(concept_h)
        
        combined_img_first = torch.cat([img_h, concept_h], dim=-1)
        combined_concept_first = torch.cat([concept_h, img_h], dim=-1)
        
        gate_img_first = torch.sigmoid(stage2_model.fusion_gate(combined_img_first))
        gate_concept_first = torch.sigmoid(stage2_model.fusion_gate(combined_concept_first))
        
        print(f"Gate stats (img_first): mean={gate_img_first.mean().item():.4f}, std={gate_img_first.std().item():.4f}, min={gate_img_first.min().item():.4f}, max={gate_img_first.max().item():.4f}")
        print(f"Gate stats (concept_first): mean={gate_concept_first.mean().item():.4f}, std={gate_concept_first.std().item():.4f}, min={gate_concept_first.min().item():.4f}, max={gate_concept_first.max().item():.4f}")
        
        variants = [
            ("img_first", "concept_gate", "sigmoid"),
            ("img_first", "img_gate", "sigmoid"),
            ("concept_first", "concept_gate", "sigmoid"),
            ("concept_first", "img_gate", "sigmoid"),
        ]
        
        print("\nTesting variants:")
        for cat_order, sum_order, gate_type in variants:
            logits = stage2_model.forward_variant(concept_emb, pooled_emb, cat_order, sum_order)
            acc = (logits.argmax(dim=1) == cell_labels).float().mean().item()
            loss = nn.CrossEntropyLoss()(logits, cell_labels).item()
            print(f"  {cat_order:15s} {sum_order:15s} -> Acc: {acc:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()

