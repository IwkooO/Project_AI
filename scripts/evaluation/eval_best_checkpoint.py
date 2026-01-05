import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from torch.utils.data import DataLoader

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase1.data import ConceptDataset
from cbm.phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from cbm.joint.train import eval_epoch, build_concept_vectors, collate_fn_joint, JointDataset
from cbm.phase2.geocells import assign_geocells, compute_offsets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths for the "best" model run
    checkpoint_dir = Path("checkpoints/stage3_joint/stage3_joint_both_scratchgeo_dim256_pos_both_gate_ct2.5_gemini")
    checkpoint_path = checkpoint_dir / "best_joint.pt"
    
    # Use paths from the run output
    phase1_run_dir = Path("checkpoints/phase1_adaptive_tau/adaptive_tau_d1_h4_dim256_pos/run_17929069")
    concept_data_dir = phase1_run_dir / "concept_data_v2"
    splits_dir = phase1_run_dir / "splits"
    test_csv = splits_dir / "dataset_test.csv"
    cached_dir = "/scratch-shared/igodzwon/Project_AI/data/6921d7831744c5356b098bf7_balanced/cached_streetclip_v2"
    
    concept_vocab_path = concept_data_dir / "concept_vocab.json"
    s2_vocab_path = concept_data_dir / "s2_cells.json"
    
    with open(concept_vocab_path, 'r') as f:
        concept_vocab = json.load(f)
    
    idx_to_concept = {int(k): v for k, v in concept_vocab.get('idx_to_concept', {}).items()}
    concept_names = [idx_to_concept[i] for i in sorted(idx_to_concept.keys())]
    num_concepts = len(concept_names)
    
    print(f"Loaded {num_concepts} concepts")
    
    # Initialize models
    phase1_model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=1024,
        concept_dim=256,
        dropout=0.45,
        mil_topk=6,
        mil_tau=0.25,
        mix_depth=1,
        mix_heads=4,
        mix_mlp_ratio=2.0,
        mix_local_kernel_size=None,
        proj_type="simple",
        use_pos_encoding=True,
        use_per_concept_tau=True,
    ).to(device)
    
    stage2_model = Stage2CrossAttentionGeoHead(
        concept_dim=256,
        patch_dim=1024,
        num_cells=1000,
        hidden_dim=512,
        mode="both",
        pooled_dim=768,
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    phase1_model.load_state_dict(checkpoint['phase1_model_state_dict'])
    stage2_model.load_state_dict(checkpoint['stage2_model_state_dict'])
    
    # Build concept vectors and adapter
    concept_vectors = build_concept_vectors(
        phase1_model,
        concept_names,
        device,
        "geolocal/StreetCLIP",
        "a street view photo containing {}"
    ).to(device)
    
    concept_adapter = ConceptEmbeddingAdapter(concept_vectors, temperature=2.5).to(device)
    if 'concept_adapter_state_dict' in checkpoint:
        concept_adapter.load_state_dict(checkpoint['concept_adapter_state_dict'])
        print("Loaded concept adapter state from checkpoint.")

    # Load test dataset
    print(f"Loading test dataset from {test_csv}...")
    base_test_ds = ConceptDataset(
        str(test_csv),
        cached_dir,
        str(concept_vocab_path),
        str(s2_vocab_path),
        split="test",
        load_pooled_embeddings=True,
    )
    
    # Load geocells from checkpoint
    geocells_path = checkpoint_dir / "phase2" / "geocells.json"
    print(f"Loading geocells from {geocells_path}...")
    with open(geocells_path, 'r') as f:
        geocell_data = json.load(f)
    centers_xyz = np.array(geocell_data['centers_xyz'])
    
    # Assign geocells to test set
    print("Assigning geocells to test set...")
    test_coords = base_test_ds._coords
    test_cell_labels = assign_geocells(test_coords, centers_xyz=centers_xyz)
    test_offsets = compute_offsets(test_coords, test_cell_labels, centers_xyz)
    
    test_ds = JointDataset(base_test_ds, test_cell_labels, test_offsets)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_fn_joint(b, test_ds.pooled_embeddings),
        num_workers=4,
    )
    
    # Criteria
    concept_criterion = nn.CrossEntropyLoss()
    cell_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()
    
    # Evaluate
    print("Running evaluation...")
    test_loss, test_concept_loss, test_cell_loss, test_offset_loss, \
    test_concept_acc1, test_concept_acc5, test_cell_acc, \
    test_mean_error, test_median_error, test_threshold_accs, _ = eval_epoch(
        phase1_model,
        stage2_model,
        concept_adapter,
        test_loader,
        device,
        concept_criterion,
        cell_criterion,
        offset_criterion,
        centers_xyz,
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Loss: {test_loss:.4f} (Concept: {test_concept_loss:.4f}, Cell: {test_cell_loss:.4f}, Offset: {test_offset_loss:.4f})")
    print(f"  Concept Accuracy - Top-1: {test_concept_acc1:.4f}")
    print(f"  Concept Accuracy - Top-5: {test_concept_acc5:.4f}")
    print(f"  Cell Accuracy: {test_cell_acc:.4f}")
    print(f"  Mean Error: {test_mean_error:.2f} km")
    print(f"  Median Error: {test_median_error:.2f} km")
    print(f"  Threshold Accuracies:")
    for threshold, acc in test_threshold_accs.items():
        print(f"    {threshold}: {acc:.4f}")

if __name__ == "__main__":
    main()

