from texttable import Texttable
from tqdm import tqdm
import json

def graph_level_reader(path):
    data = json.load(open(path))
    return data

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def get_graph_labels_and_features(graph_files):
    """

    :param graph_files:
    :return features:
    :return labels:
    """
    labels = set()
    features = set()
    for graph_file in tqdm(graph_files):
        data = json.load(open(graph_file))
        labels = labels.union(set([data["label"]]))
        features = features.union(set([val for k,v in data["features"].items() for val in v]))
    labels = {v:i for i,v in enumerate(labels)}
    features = {v:i for i,v in enumerate(features)}
    return features, labels
