import torch
from torch_sparse import spmm

class SparseGCNConv(torch.nn.Module):
    """
    Multi-scale GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = spmm(features["indices"], features["values"], features["dimensions"][0],  self.weight_matrix)
        features = spmm(normalized_adjacency_matrix["indices"], normalized_adjacency_matrix["values"], base_features.shape[0], base_features)
        return base_features
