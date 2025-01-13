"""
This script includes the components commonly used in graph-based systems. 
Multi-perspective learning structure, with utility functions and components for 
Graph analysis techniques lie in graph convolution processes alongside local message-passing methods and the utilisation of K nearest neighbours (kNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Functional

# Utility Functions

def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between two sets of points.

    Args:
        src (torch.Tensor): Source points, shape [B, N, C].
        dst (torch.Tensor): Destination points, shape [B, M, C].

    Returns:
        torch.Tensor: Squared distances, shape [B, N, M].
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Index points using the provided indices.

    Args:
        points (torch.Tensor): Input points, shape [B, N, C].
        idx (torch.Tensor): Indices to sample, shape [B, S].

    Returns:
        torch.Tensor: Indexed points, shape [B, S, C].
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz, npoint):
    """
    Perform farthest point sampling on the point cloud.

    Args:
        xyz (torch.Tensor): Input point cloud, shape [B, N, 3].
        npoint (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of sampled points, shape [B, npoint].
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    """
    Find the k-nearest neighbors.

    Args:
        nsample (int): Number of neighbors to find.
        xyz (torch.Tensor): Source points, shape [B, N, C].
        new_xyz (torch.Tensor): Query points, shape [B, M, C].

    Returns:
        torch.Tensor: Indices of k-nearest neighbors, shape [B, M, k].
    """
    dist = square_distance(xyz, new_xyz)
    return torch.topk(dist, k=nsample, dim=1, largest=False)[1].transpose(1, 2)

# Modules

class KNN_dist(nn.Module):
    """
    K-Nearest Neighbors with learned distance metric.
    """
    def __init__(self, k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1),
        )
        self.k = k

    def forward(self, F, vertices):
        """
        Perform KNN and compute weighted features.

        Args:
            F (torch.Tensor): Input features, shape [B, N, C].
            vertices (torch.Tensor): Input points, shape [B, N, 3].

        Returns:
            torch.Tensor: Aggregated features, shape [B, N, C].
        """
        id = knn(self.k, vertices, vertices)
        F = index_points(F, id)
        v = index_points(vertices, id)
        v_0 = v[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.k, 1)
        v_F = torch.cat((v_0, v, v_0 - v, torch.norm(v_0 - v, dim=-1, p=2).unsqueeze(-1)), -1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        return torch.sum(F, -2)

class View_selector(nn.Module):
    """
    Module for selecting representative views.
    """
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512 * self.s_views, 256 * self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256 * self.s_views, 40 * self.s_views)
        )

    def forward(self, F, vertices, k):
        """
        Select and refine views using the farthest point sampling and KNN.

        Args:
            F (torch.Tensor): Input features, shape [B, N, C].
            vertices (torch.Tensor): Input points, shape [B, N, 3].
            k (int): Number of neighbors for KNN.

        Returns:
            tuple: Aggregated features, scores, and new vertices.
        """
        id = farthest_point_sample(vertices, self.s_views)
        vertices1 = index_points(vertices, id)
        id_knn = knn(k, vertices, vertices1)
        F = index_points(F, id_knn)
        vertices = index_points(vertices, id_knn)
        F1 = F.transpose(1, 2).reshape(F.shape[0], k, self.s_views * F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0], k, self.s_views, 40).transpose(1, 2)
        F1_ = Functional.softmax(F_score, -3)
        F1_ = torch.max(F1_, -1)[0]
        F1_id = torch.argmax(F1_, -1)
        F1_id = Functional.one_hot(F1_id, 4).float()
        F_new = torch.mul(F1_id.unsqueeze(-1).repeat(1, 1, 1, F.shape[-1]), F).sum(-2)
        vertices_new = torch.mul(F1_id.unsqueeze(-1).repeat(1, 1, 1, 3), vertices).sum(-2)
        return F_new, F_score, vertices_new

class LocalGCN(nn.Module):
    """
    Local Graph Convolutional Network.
    """
    def __init__(self, k, n_views):
        super(LocalGCN, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)

    def forward(self, F, V):
        """
        Apply local graph convolution.

        Args:
            F (torch.Tensor): Input features, shape [B, N, C].
            V (torch.Tensor): Input points, shape [B, N, 3].

        Returns:
            torch.Tensor: Updated features, shape [B, N, C].
        """
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        return F.view(-1, self.n_views, 512)

class NonLocalMP(nn.Module):
    """
    Non-local Message Passing Module.
    """
    def __init__(self, n_view):
        super(NonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, F):
        """
        Perform non-local message passing.

        Args:
            F (torch.Tensor): Input features, shape [B, N, C].

        Returns:
            torch.Tensor: Updated features, shape [B, N, C].
        """
        F_i = F.unsqueeze(2).repeat(1, 1, self.n_view, 1)
        F_j = F.unsqueeze(1).repeat(1, self.n_view, 1, 1)
        M = self.Relation(torch.cat((F_i, F_j), -1)).sum(-2)
        F = self.Fusion(torch.cat((F, M), -1).view(-1, 2 * 512))
        return F.view(-1, self.n_view, 512)
