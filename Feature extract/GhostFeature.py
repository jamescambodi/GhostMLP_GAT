import os
import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import PointNet++ utility functions
from torch.utils.data import DataLoader

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist



def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N, dtype=torch.float32).to(device) * 1e10  # Ensure Float type
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1).float()  # Ensure dist is Float type
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint

        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

        
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
    
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import SinglePoint
from torch.cuda.amp import autocast

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("-num_class", type=int, default=40, help="Number of classes")
parser.add_argument('--val_path', default='../data/single_view_modelnet/*/test', help='Path to the test data')
parser.add_argument('--train_path', default='../data/single_view_modelnet/*/train', help='Path to the train data')
parser.add_argument('--output_data_path', default='../data/modelnet_trained_ghostfeature/', help='Output feature directory')
parser.add_argument("--workers", type=int, default=16, help='Number of DataLoader workers')
parser.add_argument('--model_path', default='./log/ghostmlp_model.pth', help='path of the pre_trained model')
parser.set_defaults(train=False)

# Function to save features
def save_features(features, scene_name, file_name, output_dir, split_name):
    """Save features to a specified directory."""
    out_file = os.path.join(output_dir, scene_name, split_name)
    os.makedirs(out_file, exist_ok=True)
    file_path = os.path.join(out_file, file_name + '.pth')
    torch.save(features.cpu(), file_path)

# Function to evaluate model
def evaluate(loader, split_name, model):
    """Evaluate the model on a dataset (train or val)."""
    mean_correct = []
    class_acc = np.zeros((args.num_class, 3))
    output_dir = args.output_data_path

    model.eval()

    with torch.no_grad():
        for data in tqdm(loader, total=len(loader), desc=f"Evaluating {split_name}"):
            points, target, scene_name, file_name = data[0], data[1], data[2], data[3]

            # Preprocess points
            points = points.transpose(1, 2).float().cuda(non_blocking=True)  # Swap dimensions 1 and 2
            target = target.cuda(non_blocking=True)

            # Forward pass
            with autocast():
                pred, _, features = model(points)

            # Save features
            file_name_no_ext = os.path.splitext(os.path.basename(file_name[-1]))[0]
            #print(f"features: {features.shape}")
            save_features(features, scene_name[-1], file_name_no_ext, output_dir, split_name)

            # Voting mechanism
            vote_pool = torch.zeros(target.shape[0], args.num_class).cuda()
            vote_pool += pred
            pred_choice = vote_pool.data.max(1)[1]

            # Accuracy calculation
            for cat in np.unique(target.cpu()):
                class_acc[cat, 0] += pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum().item()
                class_acc[cat, 1] += target[target == cat].shape[0]
            correct = pred_choice.eq(target.long().data).cpu().sum().item()
            mean_correct.append(correct / points.size(0))

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        instance_acc = np.mean(mean_correct)
        mean_class_acc = np.mean(class_acc[:, 2])

        print(f"{split_name} Instance Accuracy: {instance_acc:.4f}, Class Accuracy: {mean_class_acc:.4f}")
        return instance_acc, mean_class_acc

def count_model_parameters(model):
    # Total parameters (trainable + non-trainable)
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


if __name__ == '__main__':
    args, _ = parser.parse_known_args()

    torch.cuda.set_device(1)

    # Load datasets and dataloaders
    train_dataset = SinglePoint(args.train_path)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batchSize, num_workers=args.workers, pin_memory = True)
    val_dataset = SinglePoint(args.val_path)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batchSize, num_workers=args.workers, pin_memory = True)

    # Initialize the model
    model = GhostMLPModel(args.num_class)
    total_params, trainable_params = count_model_parameters(model)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    model = model.to('cuda:1')

    # Load the pre-trained model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    model = torch.nn.DataParallel(model, device_ids=[1])
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    print(f"Loaded model from {args.model_path}")

    # Evaluate on train and val sets
    print("Evaluating on Train Set...")
    evaluate(train_loader, "Train", model)
    print("Evaluating on Validation Set...")
    evaluate(val_loader, "Validation", model)
