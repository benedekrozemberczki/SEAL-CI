import glob
import torch
import random
import numpy as np
from layers import SAGE, MacroGCN
from tqdm import trange
from utils import get_graph_labels_and_features, graph_level_reader, hierarchical_graph_reader

class SEALTrainer(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective
    """

    def __init__(self,args):
        self.args = args
        self.hierarchical_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        self._enumerate_graphs()
        self._create_split()
        self._create_target_map()
        self._setup_macro_graph()

    def _enumerate_graphs(self):

        print("\nEnumerating graphs.\n")

        self.graphs = glob.glob(self.args.graphs + "*.json")
        features, labels = get_graph_labels_and_features(self.graphs)
        self.features = features
        self.labels = labels
        self.number_of_features = len(self.features)
        self.number_of_labels = len(self.labels)


    def _create_split(self):
        self.graph_indices = [i for i in range(len(self.graphs))]
        random.shuffle(self.graph_indices)
        self.labeled_indices = self.graph_indices[0:self.args.labeled_count]
        self.unlabeled_indices = self.graph_indices[self.args.labeled_count:]

    def _create_target_map(self):
        self.target_map = dict()
        for index in self.labeled_indices:
            path = self._concatenate_name(index)
            self.target_map[index] = graph_level_reader(path)["label"]

    def _setup_model(self):
        self.graph_level_model = SAGE(self.args, self.number_of_features, self.number_of_labels)
        self.hierarchical_model = MacroGCN(self.args, self.args.second_gcn_dimensions*self.args.second_dense_neurons, self.number_of_labels)

    def _setup_macro_graph(self):
        self.macro_graph_edges = torch.t(torch.LongTensor([[edge[0],edge[1]] for edge in self.hierarchical_graph.edges()]))

    def _transform_target(self, index):
        """
        """
        return torch.LongTensor([self.target_map[index]])

    def _transform_edges(self, raw_data):
        """
        """
        return torch.t(torch.LongTensor(raw_data["edges"]))

    def _concatenate_name(self, index):
        return self.args.graphs + str(index) + ".json"

    def _transform_features(self, raw_data):
        """
        """
        feature_matrix = np.zeros((len(raw_data["features"]), self.number_of_features))
        index_1 = [int(node) for node, features in raw_data["features"].items() for feature in features]
        index_2 = [int(self.features[feature]) for node, features in raw_data["features"].items() for feature in features]
        feature_matrix[index_1, index_2] = 1.0
        feature_matrix = torch.FloatTensor(feature_matrix)
        return feature_matrix

    def _create_batches(self):
        """
        Batching the graphs for training.
        """
        self.batches = [self.labeled_indices[i:i + self.args.batch_size] for i in range(0,len(self.labeled_indices), self.args.batch_size)]


    def _data_transform(self, raw_data):
        clean_data = dict()
        clean_data["edges"] = self._transform_edges(raw_data)
        clean_data["features"] = self._transform_features(raw_data)
        return clean_data
   
    def _score_graph_level_model(self):
        """

        """
        embeddings = []
        self.graph_level_model.eval()
        for index in self.graph_indices:
            path = self._concatenate_name(index)
            raw_data = graph_level_reader(path)
            data = self._data_transform(raw_data)
            graph_embedding, penalty, predictions = self.graph_level_model(data)
            embeddings.append(graph_embedding)
        embeddings = torch.cat(tuple(embeddings))
        embeddings = embeddings.detach()
        return embeddings

    def _create_hierarchical_target(self):
        self.hierarhical_node_indices = torch.LongTensor([node for node in range(self.hierarchical_graph.number_of_nodes())])
        self.hierarchical_mask = torch.LongTensor([0 for node in self.hierarchical_graph.nodes()])
        self.hierarchical_target = torch.LongTensor([0 for node in self.hierarchical_graph.nodes()])
        self.hierarhical_target_values = torch.LongTensor([v for k,v in self.target_map.items()])
        self.hierarhical_target_indices = torch.LongTensor([k for k,v in self.target_map.items()])
        self.hierarchical_mask[self.hierarhical_target_indices] = 1.0
        self.hierarchical_target[self.hierarhical_target_indices] = self.hierarhical_target_values

    def _fit_graph_level_model(self):
        self.graph_level_model.train()
        optimizer = torch.optim.Adam(self.graph_level_model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        for epoch in range(self.args.graph_level_epochs):
            random.shuffle(self.labeled_indices)
            self._create_batches()
            losses = 0       
            for batch in self.batches:
                accumulated_losses = 0
                optimizer.zero_grad()
                for index in batch:           
                    path = self._concatenate_name(index)
                    raw_data = graph_level_reader(path)
                    data = self._data_transform(raw_data)
                    target = self._transform_target(index)

                    _, penalty, predictions = self.graph_level_model(data)
                    loss = torch.nn.functional.nll_loss(predictions, target)+self.args.lambd*penalty
                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                accumulated_losses.backward()
                optimizer.step()

    def _fit_hierarchical_model(self):
        embeddings = self._score_graph_level_model()
        optimizer = torch.optim.Adam(self.hierarchical_model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self._create_hierarchical_target()
        for epoch in range(self.args.macro_epochs):
             optimizer.zero_grad()
             predictions = self.hierarchical_model(embeddings, self.macro_graph_edges)
             self.true_target = self.hierarchical_target[self.hierarchical_mask==1]
             predictions = predictions[self.hierarchical_mask==1]
             loss = torch.nn.functional.nll_loss(predictions, self.true_target)
             loss.backward()
             optimizer.step()
        return embeddings

    def _score_hierarchical_model(self, embeddings):
        self.hierarchical_model.eval()
        _, predictions = self.hierarchical_model(embeddings, self.macro_graph_edges).max(dim=1)
        correct = predictions[self.hierarchical_mask==0].eq(self.hierarchical_target[self.hierarchical_mask==0]).sum().item()
        normalizer = len(self.graph_indices)-len(self.labeled_indices)
        accuracy = float(correct)/float(normalizer)
        return accuracy


    def _chose_best_candidate(self, embeddings):
        predictions, indices = self.hierarchical_model(embeddings, self.macro_graph_edges).max(dim=1)
        nodes = self.hierarhical_node_indices[self.hierarchical_mask==0]
        sub_predictions = predictions[self.hierarchical_mask==0]
        sub_predictions, candidate = sub_predictions.max(dim=0)
        candidate = nodes[candidate]
        label = indices[candidate]
        return candidate, label
        
        

    def _add_node_to_labeled_ones(self, candidate, label):

        self.labeled_indices = self.labeled_indices + [candidate.item()]
        self.target_map[candidate.item()] = label.item()
        self.hierarchical_mask[candidate] = 1
        self.hierarchical_target[candidate] = label
      
    def fit(self):
        print("\nTraining started.\n")
        budget_size = trange(self.args.budget, desc='Unlabeled Accuracy: ', leave=True)
        for step in budget_size:
            self._setup_model()
            self._fit_graph_level_model()
            embeddings = self._fit_hierarchical_model()
            accuracy = self._score_hierarchical_model(embeddings)
            budget_size.set_description("Unlabeled Accuracy:%g" % round(accuracy, 4))
            candidate, label = self._chose_best_candidate(embeddings)
            self._add_node_to_labeled_ones(candidate, label)
        return embeddings

    def score(self, embeddings):
        print("\nModel scoring.\n")
        accuracy = self._score_hierarchical_model(embeddings)
        print("Unlabeled Accuracy:%g" % round(accuracy, 4))
