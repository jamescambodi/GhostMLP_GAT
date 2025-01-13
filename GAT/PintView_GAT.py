from Model import Model
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import (
    View_selector,
    LocalGCN,
    ImprovedNonLocalMP,
    GraphAttentionNetwork,
)


class DepthwiseSeparableConv(nn.Module):
    """why we use Depthwise Separable Convolution?
    --> It has fewer parameters than standard convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super(SelfAttention, self).__init__()
        # Define input/output dimensions
        self.query_conv = DepthwiseSeparableConv(in_dim, in_dim // 8)
        self.key_conv = DepthwiseSeparableConv(in_dim, in_dim // 8)
        self.value_conv = DepthwiseSeparableConv(in_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.reduce_channels = None

        # Add reduction layer if input channels do not match expected channels
        if out_dim is not None and out_dim != in_dim:
            self.reduce_channels = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        batch_size, C, width = x.size()

        # Dynamically reduce input channels if necessary
        if self.reduce_channels is not None:
            x = self.reduce_channels(x)
            C = x.size(1)

        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width)

        out = self.gamma * out + x
        return out





class PointViewGAT(Model):
    def __init__(self, name, nclasses=40, num_views=20):
        super(PointViewGAT, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.drop1 = nn.Dropout(0.5)

        vertices = [
            [1.44337567, 1.44337567, 1.44337567],
            [1.44337567, 1.44337567, -1.44337567],
            [1.44337567, -1.44337567, 1.44337567],
            [1.44337567, -1.44337567, -1.44337567],
            [-1.44337567, 1.44337567, 1.44337567],
            [-1.44337567, 1.44337567, -1.44337567],
            [-1.44337567, -1.44337567, 1.44337567],
            [-1.44337567, -1.44337567, -1.44337567],
            [0, 0.89205522, 2.3354309],
            [0, 0.89205522, -2.3354309],
            [0, -0.89205522, 2.3354309],
            [0, -0.89205522, -2.3354309],
            [2.3354309, 0, 0.89205522],
            [2.3354309, 0, -0.89205522],
            [-2.3354309, 0, 0.89205522],
            [-2.3354309, 0, -0.89205522],
            [0.89205522, 2.3354309, 0],
            [-0.89205522, 2.3354309, 0],
            [0.89205522, -2.3354309, 0],
            [-0.89205522, -2.3354309, 0],
        ]

        self.num_views_mine = 60
        self.vertices = torch.tensor(vertices).cuda()
        self.LocalGCN1 = GraphAttentionNetwork(k=4, n_views=self.num_views_mine // 3)
        self.NonLocalMP1 = ImprovedNonLocalMP(n_view=self.num_views_mine // 3)
        self.LocalGCN2 = GraphAttentionNetwork(k=4, n_views=self.num_views_mine // 4)
        self.NonLocalMP2 = ImprovedNonLocalMP(n_view=self.num_views_mine // 4)
        self.LocalGCN3 = GraphAttentionNetwork(k=4, n_views=self.num_views_mine // 6)
        self.NonLocalMP3 = ImprovedNonLocalMP(n_view=self.num_views_mine // 6)
        self.LocalGCN4 = GraphAttentionNetwork(k=4, n_views=self.num_views_mine // 12)

        self.View_selector1 = View_selector(
            n_views=self.num_views, sampled_view=self.num_views_mine // 4
        )
        self.View_selector2 = View_selector(
            n_views=self.num_views_mine // 4, sampled_view=self.num_views_mine // 6
        )
        self.View_selector3 = View_selector(
            n_views=self.num_views_mine // 6, sampled_view=self.num_views_mine // 12
        )

        self.attention = SelfAttention(512 * 4)  # Reduced number of channels
        self.cls = nn.Sequential(
            nn.Linear(512 * 4, 1024),  # Increase the number of output features
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),  # Decrease the number of output features
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.nclasses),
        )
        # print(self.cls)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        views = self.num_views
        # print(f"Input shape x: {x.shape}")
        y = x
        y = y.view((int(x.shape[0] / views), views, -1))
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)
        # print(f"Input shape y: {y.shape}")
        y = self.LocalGCN1(y, vertices)
        # sahep y [1,20,512]
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]

        z, F_score, vertices2 = self.View_selector1(y2, vertices, k=4)
        # shape z = [1,10,512]
        z = self.LocalGCN2(z, vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]
        # shape pooled_view2 [1,512]

        m, F_score_m, vertices_m = self.View_selector2(z2, vertices2, k=4)
        m = self.LocalGCN3(m, vertices_m)
        m2 = self.NonLocalMP3(m)
        pooled_view3 = torch.max(m, 1)[0]
        # pooled_view3 = pooled_view1 + pooled_view3

        w, F_score2, vertices3 = self.View_selector3(m2, vertices_m, k=4)
        w = self.LocalGCN4(w, vertices3)
        pooled_view4 = torch.max(w, 1)[0]
        # pooled_view4 = pooled_view4 + pooled_view1

        pooled_view = torch.cat((pooled_view1, pooled_view2, pooled_view3, pooled_view4), 1)

        # Add a "width" dimension for SelfAttention
        pooled_view = pooled_view.unsqueeze(2)  # Shape: [batch_size, channels, 1]

        # Apply SelfAttention
        pooled_view = self.attention(pooled_view)  # Still [batch_size, channels, 1]

        # Remove the added dimension after attention
        pooled_view = pooled_view.squeeze(2)  # Shape: [batch_size, channels]

        # Final classification layer
        pooled_view = self.cls(pooled_view)

        return pooled_view, F_score, F_score_m, F_score2
