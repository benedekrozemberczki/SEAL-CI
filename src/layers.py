import glob
import torch
import numpy as np
from utils import graph_level_reader
from torch_geometric.nn import GCNConv

class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """
    def __init__(self, args, number_of_features):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self._setup()

    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.first_gcn_dimensions)
        self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)
        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions, self.args.first_dense_neurons)
        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons, self.args.second_dense_neurons)

    def forward(self, data):
        """
        :param data: Data feed dictionary.
        :return graph_embedding:
        :return penalty: Regularization loss.
        """
        edges = data["edges"]
        features = data["features"]
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1),dim=0)
        graph_embedding = torch.mm(torch.t(attention), node_features_2)
        graph_embedding = graph_embedding.view(1,-1)
        penalty = torch.mm(torch.t(attention),attention)-torch.eye(self.args.second_dense_neurons)
        penalty = torch.sum(torch.norm(penalty, p=2, dim=1))
        return graph_embedding, penalty

class MacroGCN(torch.nn.Module):

    def __init__(self, args, number_of_features, number_of_labels):
        super(MacroGCN, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self._setup()

    def _setup(self):
        self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.macro_gcn_dimensions)
        self.graph_convolution_2 = GCNConv(self.args.macro_gcn_dimensions, self.number_of_labels)
        
    def forward(self, features, edges):
        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)
        predictions = torch.nn.functional.log_softmax(node_features_2,dim=1)
        return predictions

class SEAL(torch.nn.Module):
     
    def __init__(self, args, number_of_features, number_of_labels): 
        super(SEAL, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self._setup()

    def _setup(self):
        self.graph_level_model = SAGE(self.args, self.number_of_features)
        self.hierarchical_model = MacroGCN(self.args, self.args.second_gcn_dimensions*self.args.second_dense_neurons, self.number_of_labels)

    def forward(self, graphs, macro_edges):
        embeddings = []
        penalties = 0
        for graph in graphs:
            embedding, penalty = self.graph_level_model(graph)
            embeddings.append(embedding)
            penalties = penalties + penalty
        embeddings = torch.cat(tuple(embeddings))
        penalties = penalties/len(graphs)
        predictions = self.hierarchical_model(embeddings, macro_edges)
        return predictions, penalties
