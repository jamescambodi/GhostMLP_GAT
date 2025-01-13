"""
This code utilises the PointViewGCNs model, a network based on graphs created for multi-view learning.  
The system utilises both nonlocal graph convolutional layers to analyse perspectives in addition to a sequence of 
The system employs view selectors to select and improve input perspectives in its design effectively.  
spatial relationships and produce a final classification output.
"""

import torch
import torch.nn as nn
from utils import View_selector, LocalGCN, NonLocalMP
from Model import Model  # Base model class for saving and loading checkpoints

class PointViewGCN(Model):
    """
    Implements the PointViewGCN model for multi-view classification tasks.
    
    Attributes:
        nclasses (int): Number of output classes for classification.
        num_views (int): Number of input views.
        drop1 (nn.Dropout): Dropout layer for regularization.
        vertices (torch.Tensor): Predefined vertices for graph operations.
        LocalGCN1-4 (LocalGCN): Local graph convolutional layers for feature extraction.
        NonLocalMP1-3 (NonLocalMP): Non-local message passing layers for feature refinement.
        View_selector1-3 (View_selector): Layers for selecting a subset of views.
        cls (nn.Sequential): Fully connected layers for classification.
    """
    def __init__(self, name, nclasses=40, num_views=20):
        """
        Initialize the PointViewGCN model.

        Args:
            name (str): Name of the model for checkpoint organization.
            nclasses (int): Number of output classes for classification. Default is 40.
            num_views (int): Number of input views. Default is 20.
        """
        super(PointViewGCN, self).__init__(name)
        self.nclasses = nclasses
        self.num_views = num_views
        self.drop1 = nn.Dropout(0.5)  # Dropout for regularization

        # Predefined vertices for graph-based processing
        vertices = [[1.44337567, 1.44337567, 1.44337567], [1.44337567, 1.44337567, -1.44337567], [1.44337567, -1.44337567, 1.44337567], 
                    [1.44337567, -1.44337567, -1.44337567], [-1.44337567, 1.44337567, 1.44337567], [-1.44337567, 1.44337567, -1.44337567], 
                    [-1.44337567, -1.44337567, 1.44337567], [-1.44337567, -1.44337567, -1.44337567], [0, 0.89205522, 2.3354309], 
                    [0, 0.89205522, -2.3354309], [0, -0.89205522, 2.3354309], [0, -0.89205522, -2.3354309], [2.3354309, 0, 0.89205522], 
                    [2.3354309, 0, -0.89205522], [-2.3354309, 0, 0.89205522], [-2.3354309, 0, -0.89205522], [0.89205522, 2.3354309, 0], 
                    [-0.89205522, 2.3354309, 0], [0.89205522, -2.3354309, 0], [-0.89205522, -2.3354309, 0]]

        self.num_views_mine = 60  # Number of views used internally
        self.vertices = torch.tensor(vertices).cuda()  # Move vertices to GPU

        # Define LocalGCN and NonLocalMP layers
        self.LocalGCN1 = LocalGCN(k=4, n_views=self.num_views_mine // 3)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views_mine // 3)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views_mine // 4)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views_mine // 4)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views_mine // 6)
        self.NonLocalMP3 = NonLocalMP(n_view=self.num_views_mine // 6)
        self.LocalGCN4 = LocalGCN(k=4, n_views=self.num_views_mine // 12)

        # Define View_selector layers
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views_mine // 4)
        self.View_selector2 = View_selector(n_views=self.num_views_mine // 4, sampled_view=self.num_views_mine // 6)
        self.View_selector3 = View_selector(n_views=self.num_views_mine // 6, sampled_view=self.num_views_mine // 12)

        # Define classification layers
        self.cls = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.nclasses)
        )

        # Initialize weights for linear and convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size * num_views, features].

        Returns:
            tuple: The classification logits and feature scores at different stages.
        """
        views = self.num_views
        y = x.view((int(x.shape[0] / views), views, -1))  # Reshape input into [batch, views, features]
        vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)  # Expand vertices for batch processing

        # First stage: Local GCN and Non-local MP
        y = self.LocalGCN1(y, vertices)
        y2 = self.NonLocalMP1(y)
        pooled_view1 = torch.max(y, 1)[0]  # Max pooling for first stage

        # Second stage: View selection and Local GCN
        z, F_score, vertices2 = self.View_selector1(y2, vertices, k=4)
        z = self.LocalGCN2(z, vertices2)
        z2 = self.NonLocalMP2(z)
        pooled_view2 = torch.max(z, 1)[0]  # Max pooling for second stage

        # Third stage: Further refinement
        m, F_score_m, vertices_m = self.View_selector2(z2, vertices2, k=4)
        m = self.LocalGCN3(m, vertices_m)
        m2 = self.NonLocalMP3(m)
        pooled_view3 = torch.max(m, 1)[0]  # Max pooling for third stage

        # Fourth stage: Final refinement
        w, F_score2, vertices3 = self.View_selector3(m2, vertices_m, k=4)
        w = self.LocalGCN4(w, vertices3)
        pooled_view4 = torch.max(w, 1)[0]  # Max pooling for fourth stage

        # Concatenate pooled features from all stages
        pooled_view = torch.cat((pooled_view1, pooled_view2, pooled_view3, pooled_view4), 1)
        pooled_view = self.cls(pooled_view)  # Classification layers

        return pooled_view, F_score, F_score_m, F_score2  # Output logits and feature scores
