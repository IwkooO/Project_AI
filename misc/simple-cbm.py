import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

CSV_PATH = "/scratch-shared/pnair/Project_AI/data/6906237dc7731161a37282b2/export/dataset_export.csv"


class GeoGuessrCSV(Dataset):
    """
    Dataset that loads samples from dataset_export.csv

    Expected columns in CSV:
        image_path   (absolute or relative path to image file)
        concept_idx  (int)
        country_idx  (int)
        lat_norm     (float, normalized latitude)
        lng_norm     (float, normalized longitude)
    """

    def __init__(self, csv_path, img_size=224):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # infer number of concepts and countries
        self.num_concepts = int(self.df["concept_idx"].max()) + 1
        self.num_countries = int(self.df["country_idx"].max()) + 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        concept_idx = int(row["concept_idx"])
        country_idx = int(row["country_idx"])

        coord = torch.tensor(
            [float(row["lat_norm"]), float(row["lng_norm"])],
            dtype=torch.float32,
        )

        # one hot concept target
        concept_target = torch.zeros(self.num_concepts, dtype=torch.float32)
        concept_target[concept_idx] = 1.0

        return image, concept_target, country_idx, coord


class SimpleGeoCBM(nn.Module):
    """
    Simple concept bottleneck model:
      backbone -> concept logits
      concepts -> country
      concepts -> coordinates
    """

    def __init__(self, num_concepts, num_countries, pretrained=True):
        super().__init__()

        backbone = models.resnet18(pretrained=pretrained)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.concept_layer = nn.Linear(feat_dim, num_concepts)
        self.country_head = nn.Linear(num_concepts, num_countries)
        self.coord_head = nn.Linear(num_concepts, 2)

    def forward(self, x):
        feats = self.backbone(x)

        concept_logits = self.concept_layer(feats)          # (B, C)
        concept_probs = torch.sigmoid(concept_logits)       # (B, C)

        country_logits = self.country_head(concept_probs)   # (B, num_countries)
        coord_pred = self.coord_head(concept_probs)         # (B, 2)

        return concept_logits, country_logits, coord_pred


def train(
    csv_path=CSV_PATH,
    batch_size=32,
    num_epochs=5,
    lr=1e-4,
    coord_loss_weight=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = GeoGuessrCSV(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleGeoCBM(
        num_concepts=dataset.num_concepts,
        num_countries=dataset.num_countries,
        pretrained=True,
    ).to(device)

    # losses
    concept_criterion = nn.BCEWithLogitsLoss()
    country_criterion = nn.CrossEntropyLoss()
    coord_criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_concept_loss = 0.0
        running_country_loss = 0.0
        running_coord_loss = 0.0

        for i, (images, concept_targets, country_idx, coords) in enumerate(dataloader):
            images = images.to(device)
            concept_targets = concept_targets.to(device)
            country_idx = country_idx.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()

            concept_logits, country_logits, coord_pred = model(images)

            loss_concept = concept_criterion(concept_logits, concept_targets)
            loss_country = country_criterion(country_logits, country_idx)
            loss_coord = coord_criterion(coord_pred, coords)

            loss = loss_concept + loss_country + coord_loss_weight * loss_coord

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_concept_loss += loss_concept.item()
            running_country_loss += loss_country.item()
            running_coord_loss += loss_coord.item()

            if (i + 1) % 10 == 0:
                n = 10
                print(
                    f"Epoch {epoch + 1} "
                    f"Iter {i + 1} "
                    f"Total {running_loss / n:.4f} "
                    f"Concept {running_concept_loss / n:.4f} "
                    f"Country {running_country_loss / n:.4f} "
                    f"Coord {running_coord_loss / n:.4f}"
                )
                running_loss = 0.0
                running_concept_loss = 0.0
                running_country_loss = 0.0
                running_coord_loss = 0.0

    print("Training finished")
    return model


if __name__ == "__main__":
    train()
