import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run SEAL-CI/SEAL-AI.")

    parser.add_argument("--graphs",
                        nargs = "?",
                        default = "./input/graphs/",
	                help = "Folder with training graph pair jsons.")

    parser.add_argument("--hierarchical-graph",
                        nargs = "?",
                        default = "./input/synthetic_edges.csv",
	                help = "Folder with testing graph pair jsons.")

    parser.add_argument("--labeled-count",
                        type = int,
                        default = 50,
	                help = "Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--budget",
                        type = int,
                        default = 20,
	                help = "Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--first-gcn-dimensions",
                        type = int,
                        default = 32,
	                help = "Filters (neurons) in 1st convolution. Default is 32.")

    parser.add_argument("--second-gcn-dimensions",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--macro-gcn-dimensions",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in 2nd convolution. Default is 32.")

    parser.add_argument("--first-dense-neurons",
                        type = int,
                        default = 16,
	                help = "Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--second-dense-neurons",
                        type = int,
                        default = 8,
	                help = "Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 8,
	                help = "Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--macro-epochs",
                        type = int,
                        default = 50,
	                help = "Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--graph-level-epochs",
                        type = int,
                        default = 5,
	                help = "Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.001,
	                help = "Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 5*10**-5,
	                help = "Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--lambd",
                        type = float,
                        default = 10**-4,
	                help = "Adam weight decay. Default is 5*10^-4.")

    return parser.parse_args()
