import torch
import random
from layers import SEAL
from tqdm import trange
from utils import hierarchical_graph_reader, GraphDatasetGenerator 

class SEALCITrainer(object):
    """
    Semi-Supervised Graph Classification: A Hierarchical Graph Perspective Cautious Iteration model.
    """
    def __init__(self,args):
        """
        Creating dataset, doing dataset split, creating target and node index vectors.
        :param args: Arguments object.
        """
        self.args = args
        self.macro_graph = hierarchical_graph_reader(self.args.hierarchical_graph)
        self.dataset_generator = GraphDatasetGenerator(self.args.graphs)
        self._setup_macro_graph()
        self._create_split()
        self._create_labeled_target()
        self._create_node_indices()

    def _setup_model(self):
        """
        Creating a SEAL model.
        """
        self.model = SEAL(self.args, self.dataset_generator.number_of_features, self.dataset_generator.number_of_labels)

    def _setup_macro_graph(self):
        """
        Creating an edge list for the hierarchical graph.
        """
        self.macro_graph_edges = [[edge[0],edge[1]] for edge in self.macro_graph.edges()]
        self.macro_graph_edges = torch.t(torch.LongTensor(self.macro_graph_edges))

    def _create_split(self):
        """
        Creating a labeled-unlabeled split.
        """
        graph_indices = [index for index in range(len(self.dataset_generator.graphs))]
        random.shuffle(graph_indices)
        self.labeled_indices = graph_indices[0:self.args.labeled_count]
        self.unlabeled_indices = graph_indices[self.args.labeled_count:]

    def _create_labeled_target(self):
        """
        Creating a mask for labeled instances and a target for them.
        """
        self.labeled_mask = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        self.labeled_target = torch.LongTensor([0 for node in self.macro_graph.nodes()])
        self.labeled_mask[torch.LongTensor(self.labeled_indices)] = 1
        self.labeled_target[torch.LongTensor(self.labeled_indices)] = self.dataset_generator.target[torch.LongTensor(self.labeled_indices)]

    def _create_node_indices(self):
        """
        Creating an index of nodes.
        """
        self.node_indices = [index for index in range(self.macro_graph.number_of_nodes())]
        self.node_indices = torch.LongTensor(self.node_indices)

    def fit_a_single_model(self):
        """
        Fitting a single SEAL model.
        """
        self._setup_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)      
        for epoch in range(self.args.epochs):
            optimizer.zero_grad()
            predictions, penalty = self.model(self.dataset_generator.graphs, self.macro_graph_edges)
            loss = torch.nn.functional.nll_loss(predictions[self.labeled_mask==1], self.labeled_target[self.labeled_mask==1]) + self.args.gamma*penalty
            loss.backward()
            optimizer.step()

    def score_a_single_model(self):
        """
        Scoring the SEAL model.
        """
        self.model.eval()
        predictions, penalty = self.model(self.dataset_generator.graphs, self.macro_graph_edges)
        scores, prediction_indices = predictions.max(dim=1)
        correct = prediction_indices[self.labeled_mask==0].eq(self.labeled_target[self.labeled_mask==0]).sum().item()
        normalizer = prediction_indices[self.labeled_mask==0].shape[0]
        accuracy = float(correct)/float(normalizer)
        return scores, prediction_indices, accuracy
            
    def _choose_best_candidate(self, predictions, indices):
        """
        Choosing the best candidate based on predictions.
        :param predictions: Scores.
        :param indices: Vector of likely labels.
        :return candidate: Node chosen.
        :return label: Label of node.
        """
        nodes = self.node_indices[self.labeled_mask==0]
        sub_predictions = predictions[self.labeled_mask==0]
        sub_predictions, candidate = sub_predictions.max(dim=0)
        candidate = nodes[candidate]
        label = indices[candidate]
        return candidate, label

    def _update_target(self, candidate, label):
        self.labeled_mask[candidate] = 1
        self.labeled_target[candidate] = label      

    def fit(self):
        """
        Training models sequentially.
        """
        print("\nTraining started.\n")
        budget_size = trange(self.args.budget, desc='Unlabeled Accuracy: ', leave=True)
        for step in budget_size:
            self.fit_a_single_model()
            scores, prediction_indices, accuracy = self.score_a_single_model()
            candidate, label = self._choose_best_candidate(scores, prediction_indices)
            self._update_target(candidate, label)
            budget_size.set_description("Unlabeled Accuracy:%g" % round(accuracy, 4))

    def score(self):
        """
        Scoring the model.
        """
        print("\nModel scoring.\n")
        scores, prediction_indices, accuracy = self.score_a_single_model()
        print("Unlabeled Accuracy:%g" % round(accuracy, 4))
