import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HaversineLoss(nn.Module):
    """
    Calculates the Haversine distance (Great Circle Distance) between two points on Earth.
    Input: (Lat, Lon) in Degrees.
    Output: Distance in Kilometers.
    """
    def __init__(self):
        super().__init__()
        self.R = 6371.0  # Earth radius in kilometers

    def forward(self, pred_coords, target_coords):
        """
        Args:
            pred_coords: Tensor of shape (batch_size, 2) containing (lat, lon) in degrees.
            target_coords: Tensor of shape (batch_size, 2) containing (lat, lon) in degrees.
        Returns:
            Average distance in kilometers.
        """
        # Convert degrees to radians
        pred_rad = torch.deg2rad(pred_coords)
        target_rad = torch.deg2rad(target_coords)

        lat1, lon1 = pred_rad[:, 0], pred_rad[:, 1]
        lat2, lon2 = target_rad[:, 0], target_rad[:, 1]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        # Clamp a to [0, 1] to avoid numerical instability in sqrt/asin
        a = torch.clamp(a, min=0.0, max=1.0)
        c = 2 * torch.asin(torch.sqrt(a))

        distance = self.R * c
        return distance.mean()

class CBMLoss(nn.Module):
    """
    Joint loss for Concept Bottleneck Model:
    Loss = lambda_coords * HaversineLoss + lambda_country * CE + lambda_concepts * CE
    """
    def __init__(self, lambda_coords=1.0, lambda_country=1.0, lambda_concepts=0.5):
        super().__init__()
        self.haversine = HaversineLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        self.lambda_coords = lambda_coords
        self.lambda_country = lambda_country
        self.lambda_concepts = lambda_concepts

    def forward(self, 
                concept_logits, concept_targets, 
                country_logits, country_targets, 
                pred_coords, target_coords):
        
        # 1. Concept Loss
        concept_loss = self.cross_entropy(concept_logits, concept_targets)
        
        # 2. Country Loss
        country_loss = self.cross_entropy(country_logits, country_targets)
        
        # 3. Coordinate Loss (Haversine)
        # Calculate raw distance in km
        haversine_dist = self.haversine(pred_coords, target_coords)
        
        # Normalize: Divide by 1000.0 so that 1.0 loss ~= 1000km error
        # This balances the magnitude with CrossEntropy (~0.5 - 5.0)
        coord_loss = haversine_dist / 1000.0
        
        # Weighted Sum
        total_loss = (self.lambda_coords * coord_loss) + \
                     (self.lambda_country * country_loss) + \
                     (self.lambda_concepts * concept_loss)
                     
        return total_loss, {
            "loss": total_loss,
            "concept_loss": concept_loss,
            "country_loss": country_loss,
            "coord_loss": coord_loss, # Normalized loss
            "haversine_dist": haversine_dist # Raw distance in km
        }

