import glob
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from utils import get_graph_labels_and_features, graph_level_reader

class SEAL(torch.nn.Module):

    def __init__(self, args, number_of_features, number_of_labels):
        super(SEAL, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self._setup()

    def _setup(self):
        self.graph_convolution_1 = GCNConv(self.number_of_features, self.args.first_gcn_dimensions)
        self.graph_convolution_2 = GCNConv(self.args.first_gcn_dimensions, self.args.second_gcn_dimensions)
        self.fully_connected_1 = torch.nn.Linear(self.args.second_gcn_dimensions, self.args.first_dense_neurons)
        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons, self.args.second_dense_neurons)
        self.last_fully_connected = torch.nn.Linear(self.args.second_gcn_dimensions*self.args.second_dense_neurons, self.number_of_labels)

    def forward(self, data):
        edges = data["edges"]
        features = data["features"]

        node_features_1 = torch.nn.functional.relu(self.graph_convolution_1(features, edges))
        node_features_2 = self.graph_convolution_2(node_features_1, edges)

        abstract_features_1 = torch.tanh(self.fully_connected_1(node_features_2))
        attention = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1),dim=1)
        graph_embedding = torch.mm(torch.t(attention), node_features_2)
        graph_embedding = graph_embedding.view(1,-1)

        penalty = torch.mm(torch.t(attention),attention)-torch.eye(self.args.second_dense_neurons)
        penalty = torch.sum(torch.norm(penalty, p=2, dim=1))
        predictions = self.last_fully_connected(graph_embedding)
        predictions = torch.nn.functional.log_softmax(predictions,dim=1)
        return graph_embedding, penalty, predictions

class SEALTrainer(object):


    def __init__(self,args):
        self.args = args
        self._enumerate_graphs()
        self._setup_model()

    def _enumerate_features_and_labels(self):
        graph_files = self.training_graphs + self.testing_graphs
        features, labels = get_graph_labels_and_features(graph_files)
        self.features = features
        self.labels = labels
        self.number_of_features = len(self.features)
        self.number_of_labels = len(self.labels)

    def _enumerate_graphs(self):
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        self._enumerate_features_and_labels()


    def _setup_model(self):
        self.model = SEAL(self.args, self.number_of_features, self.number_of_labels)

    def _transform_target(self, raw_data):
        """
        """
        return torch.FloatTensor([0.0 if i != raw_data["label"] else 1.0 for i in range(self.number_of_labels)])

    def _transform_edges(self, raw_data):
        """
        """
        return torch.t(torch.LongTensor(raw_data["edges"]))

    def _transform_features(self, raw_data):
        """
        """
        feature_matrix = np.zeros((len(raw_data["features"]), self.number_of_features))
        index_1 = [int(node) for node, features in raw_data["features"].items() for feature in features]
        index_2 = [int(self.features[feature]) for node, features in raw_data["features"].items() for feature in features]
        feature_matrix[index_1, index_2] = 1.0
        feature_matrix = torch.FloatTensor(feature_matrix)
        return feature_matrix

    def _data_transform(self, raw_data):
        clean_data = dict()
        clean_data["target"] = self._transform_target(raw_data)
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["features"] = self._transform_features(raw_data)
        return clean_data

    def fit(self):
        print("Model fit.")
        for graph in tqdm(self.training_graphs):
            raw_data = graph_level_reader(graph)
            data = self._data_transform(raw_data)
            graph_embedding, penalty, predictions = self.model(data)

    def score(self):
        print("Model score.")
