import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import GhostMLPSetAbstractionMsg, GhostMLPSetAbstraction

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0

        self.normal_channel = normal_channel

        # Replace PointNetSetAbstractionMsg with GhostMLPSetAbstractionMsg
        self.sa1 = GhostMLPSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = GhostMLPSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = GhostMLPSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # Pass through GhostMLP abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Flatten the features for the fully connected layers
        x = l3_points.view(B, 1024)
        x_feature = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x2 = self.drop2(F.relu(self.bn2(self.fc2(x_feature))))
        x = self.fc3(x2)
        x = F.log_softmax(x, -1)
        return x, l3_points, x_feature

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        # Compute the loss
        total_loss = F.nll_loss(pred, target)
        return total_loss   
