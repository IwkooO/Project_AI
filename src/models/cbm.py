
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptHead(nn.Module):
    """
    Predicts concept probabilities from image embeddings.
    Produces a richer hidden embedding to support contrastive/metric losses.
    """
    def __init__(self, input_dim=768, num_concepts=100, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.second = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, num_concepts)
    
    def forward(self, x):
        h = self.pre(x)
        gated = torch.sigmoid(self.gate(x))
        h = h + gated * h
        h2 = self.second(h)
        hidden = h + h2  # residual
        logits = self.output_layer(hidden)
        return logits, hidden


class GeoHead(nn.Module):
    """
    Predicts S2 cell and offset from image embeddings + concept probabilities.
    
    Coarse: [z, c_probs] -> MLP -> M cells
    Fine:   [z, c_probs] -> MLP -> M * 2 offsets
    """
    def __init__(self, input_dim=768, num_concepts=100, num_cells=1000, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Input is image embedding + concept probabilities
        combined_dim = input_dim + num_concepts
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Coarse Head (Cell Classification)
        self.coarse_head = nn.Linear(hidden_dim, num_cells)
        
        # Fine Head (Offset Regression)
        # Predicts (d_lat, d_lng) for each cell
        self.fine_head = nn.Linear(hidden_dim, num_cells * 2)
        
        self.num_cells = num_cells

    def forward(self, z, c_probs):
        # z: [B, 768]
        # c_probs: [B, K]
        
        x = torch.cat([z, c_probs], dim=1)
        features = self.feature_net(x)
        
        # Coarse logits: [B, M]
        cell_logits = self.coarse_head(features)
        
        # Fine offsets: [B, M, 2]
        offsets = self.fine_head(features).view(-1, self.num_cells, 2)
        
        return cell_logits, offsets


class CBM(nn.Module):
    """
    Concept Bottleneck Model for Geolocation.
    Phase 1: Train ConceptHead
    Phase 2: Freeze ConceptHead, Train GeoHead
    """
    def __init__(self, num_concepts, num_cells, input_dim=768, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.concept_head = ConceptHead(input_dim, num_concepts, hidden_dim, dropout)
        self.geo_head = GeoHead(input_dim, num_concepts, num_cells, hidden_dim, dropout)
        
    def forward(self, z):
        # 1. Concept Prediction
        c_logits, c_hidden = self.concept_head(z)
        c_probs = torch.softmax(c_logits, dim=1)
        
        # 2. Geo Prediction (using soft concept probs)
        # Detach c_probs if we want to stop gradients from geo loss to concept head (Phase 2 style)
        # But usually we can train joint or freeze concept head via optimizer
        cell_logits, offsets = self.geo_head(z, c_probs)
        
        return c_logits, cell_logits, offsets, c_hidden


