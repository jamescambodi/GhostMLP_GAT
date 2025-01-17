import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F

# Import PointNet++ utility functions
from torch.utils.data import DataLoader

import torch.optim as optim
from dataloader import SinglePoint
from tqdm import tqdm
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction
        
# Define GhostModule
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, kernel_size=1, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        init_channels = max(1, int(out_channels / ratio))
        cheap_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, cheap_channels, dw_size, stride=1, padding=dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm1d(cheap_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if x2.shape[1] < x1.shape[1]:  # Handle edge cases
            padding = x1.shape[1] - x2.shape[1]
            x2 = F.pad(x2, (0, 0, 0, padding))

        out = torch.cat([x1, x2], dim=1)
        return out[:, :x.shape[1], :]  # Ensure output shape matches


# Define GhostMLPModel
class GhostMLPModel(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(GhostMLPModel, self).__init__()
        in_channel = 3 if normal_channel else 0

        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.encoder = nn.Sequential(
            GhostModule(1024, 512),
            nn.Dropout(0.4),
            GhostModule(512, 512),
            nn.Dropout(0.5),
        )
        
        self.fc3 = nn.Linear(512, num_class)  # Classification layer

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Flatten the global features
        x = l3_points.view(B, 1024)

        # Feature encoding
        features = self.encoder(x.unsqueeze(-1)).squeeze(-1)  # [B, 3] after projection

        # Final classification output
        logits = F.log_softmax(self.fc3(features), dim=-1)

        return logits, l3_points, features  # Output logits, global features, and reduced features

# Normalize point cloud
def pc_normalize(pc):
    centroid = pc.mean(axis=0)
    pc -= centroid
    m = (pc**2).sum(axis=1).max()**0.5
    pc /= m
    return pc

# Custom collate function for DataLoader
def collate_fn(batch):
    points = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    points = [pc_normalize(point_set) for point_set in points]
    points = np.array(points)
    points = torch.tensor(points, dtype=torch.float32).permute(0, 2, 1)  # [B, 3, N]
    labels = torch.tensor(labels, dtype=torch.long)
    return points, labels

# Training loop
def train(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for points, labels in tepoch:
                points, labels = points.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                logits, _, _ = model(points)

                # Compute loss
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {100.0 * correct / total:.2f}%")


# Main script
if __name__ == "__main__":
    # Parameters
    num_classes = 40  # Based on the classnames in the dataloader
    num_epochs = 25
    batch_size = 32
    learning_rate = 0.001
    save_path = "ghostmlp_model.pth"
    dataset_root = "../data/single_view_modelnet/*/train"  # Update this to your dataset path

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    train_dataset = SinglePoint(dataset_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model, loss, optimizer
    model = GhostMLPModel(num_class=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    print(f"Memory required for parameters (MB): {total_params * 4 / (1024**2)}")

    print(f"Before .to(device): Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Before .to(device): Reserved memory: {torch.cuda.memory_reserved() / 1e6} MB")
    model = model.to(device)
    print(f"After .to(device): Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"After .to(device): Reserved memory: {torch.cuda.memory_reserved() / 1e6} MB")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
