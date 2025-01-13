import torch
import torch.nn as nn
import torch.nn.functional as Functional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """This function is used to calculate the square distance between two sets of points.

    Args:
        src (torch.Tensor): source points, [B, N, C]
        dst (torch.Tensor): destination points, [B, M, C]

    Returns:
        _type_: _description_
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    This function is used for selecting specific points from a point cloud based on the input index.

    Input:
        points: input points data, [B, N, C], where B is batch size, N is number of points, C is number of channels
        idx: sample index data, [B, S], where B is batch size and S is number of samples
    Return:
        new_points:, indexed points data, [B, S, C], where B is batch size, S is number of samples, C is number of channels
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )

    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int):
    """
    The function performs farthest point sampling (FPS),
    which is a technique to select a subset of points from a point cloud such that the selected points are as far apart from each other as possible.
    This is useful for reducing the number of points while preserving the overall structure and distribution of the point cloud.

    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
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
    """The function is used for performing k-nearest neighbor search.

    Args:
        nsample (int): number of neighbors
        xyz (torch.Tensor): input points data, [B, N, C]
        new_xyz (torch.Tensor): query points data, [B, M, C]

    Returns:
        torch.Tensor: group index, [B, M, nsample]
    """
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist, k=nsample, dim=1, largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id


# OLD MODEL
# class KNN_dist(nn.Module):
#     def __init__(self, k):
#         super(KNN_dist, self).__init__()
#         self.R = nn.Sequential(
#             nn.Linear(10, 1),
#         )
#         self.k = k

#     def forward(self, F, vertices):
#         id = knn(self.k, vertices, vertices)
#         F = index_points(F, id)
#         v = index_points(vertices, id)
#         v_0 = v[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.k, 1)
#         v_F = torch.cat(
#             (v_0, v, v_0 - v, torch.norm(v_0 - v, dim=-1, p=2).unsqueeze(-1)), -1
#         )
#         v_F = self.R(v_F)
#         F = torch.mul(v_F, F)
#         F = torch.sum(F, -2)
#         return F


class KNN_dist(nn.Module):
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
        id = knn(self.k, vertices, vertices)
        F = index_points(F, id)
        v = index_points(vertices, id)
        v_0 = v[:, :, 0, :].unsqueeze(-2).repeat(1, 1, self.k, 1)
        v_F = torch.cat((v_0, v, v_0 - v, torch.norm(v_0 - v, dim=-1, p=2).unsqueeze(-1)), -1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F, -2)

        return F


class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512 * self.s_views, 256 * self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256 * self.s_views, 40 * self.s_views),
        )

    def forward(self, F, vertices, k):
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
        F1_id_v = F1_id.unsqueeze(-1).repeat(1, 1, 1, 3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F, F).sum(-2)
        vertices_new = torch.mul(F1_id_v, vertices).sum(-2)
        return F_new, F_score, vertices_new


# OLD MODEL
class LocalGCN(nn.Module):
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
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class GraphAttentionNetwork(nn.Module):
    def __init__(self, k, n_views):
        super(GraphAttentionNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
        self.attention = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, F, V):
        F = self.KNN(F, V)

        # Ensure F has four dimensions
        if len(F.shape) == 3:
            # If F is missing the `k` dimension, add it with k=1
            F = F.unsqueeze(2)

        # Unpack dimensions
        batch_size, n_views, k, feature_dim = F.shape

        # Flatten for convolution
        F = F.view(-1, feature_dim)
        F = self.conv(F)

        # Reshape back to [batch_size, n_views, feature_dim]
        F = F.view(batch_size, n_views, feature_dim)

        # Attention mechanism
        attention_weights = self.attention(F).squeeze(-1)
        ##print(f"attention_weights.shape: {attention_weights.shape}")
        attention_weights = Functional.softmax(attention_weights, dim=-1)
        ##print(f"attention_weights after softmax: {attention_weights.shape}")

        # Apply attention
        F = F * attention_weights.unsqueeze(-1)
        ##print(f"F.shape after attention weighting: {F.shape}")

        # Second convolution
        F = F.view(-1, feature_dim)
        F = self.conv(F)

        # Reshape back to [batch_size, n_views, feature_dim]
        F = F.view(batch_size, n_views, feature_dim)
        ##print(f"F.shape after final reshaping: {F.shape}")

        return F







# OLD MODEL
class OldNonLocalMP(nn.Module):
    def __init__(self, n_view):
        super(OldNonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
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
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M, -2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F


class NonLocalMP(nn.Module):
    def __init__(self, n_view):
        super(NonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Attention = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, F):
        # Unsqueeze and repeat to align dimensions
        F_i = F.unsqueeze(1).repeat(1, self.n_view, 1, 1)  # Shape: [128, 20, 20, 512]
        F_j = F.unsqueeze(2).repeat(1, 1, self.n_view, 1)  # Shape: [128, 20, 20, 512]
        print(f"F_i.shape: {F_i.shape}, F_j.shape: {F_j.shape}")
        
        # Concatenate along the last dimension
        M = torch.cat((F_i, F_j), dim=3)  # Shape: [128, 20, 20, 1024]
        M = self.Relation(M)  # Apply Relation network
        M = torch.sum(M, dim=2)  # Sum along the 3rd dimension
        F = torch.cat((F, M), dim=2)  # Concatenate features
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)  # Apply Fusion network
        F = F.view(-1, self.n_view, 512)
        
        # Attention mechanism
        attention_weights = self.Attention(F).squeeze(-1)
        attention_weights = attention_weights.unsqueeze(1)
        F = torch.matmul(attention_weights, F).squeeze(1)
        return F


class NonLocalMP(nn.Module):
    def __init__(self, n_view):
        super(NonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
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
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M, -2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F



class ImprovedNonLocalMP(nn.Module):
    def __init__(self, n_view):
        super(ImprovedNonLocalMP, self).__init__()
        self.n_view = n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Attention = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M, -2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)

        # Attention mechanism
        attention_weights = self.Attention(F).squeeze(-1)
        attention_weights = Functional.softmax(attention_weights, dim=1)  # Normalize attention
        F = torch.mul(attention_weights.unsqueeze(-1), F)
        ##print(f"Final F.shape after attention: {F.shape}")
        return F
