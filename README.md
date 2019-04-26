SEAL
============================================
An implementation of "Semi-Supervised Graph Classification: A Hierarchical Graph Perspective" (WWW 2019)
<p align="center">
  <img width="800" src="seal.jpg">
</p>
<p align="justify">
Node classification and graph classification are two graph learning problems that predict the class label of a node and the class label of a graph respectively. A node of a graph usually represents a real-world entity, e.g., a user in a social network, or a protein in a protein-protein interaction network. In this work, we consider a more challenging but practically useful setting, in which a node itself is a graph instance. This leads to a hierarchical graph perspective which arises in many domains such as social network, biological network and document collection. For example, in a social network, a group of people with shared interests forms a user group, whereas a number of user groups are interconnected via interactions or common members. We study the node classification problem in the hierarchical graph where a `node' is a graph instance, e.g., a user group in the above example. As labels are usually limited in real-world data, we design two novel semi-supervised solutions named \underline{SE}mi-supervised gr\underline{A}ph c\underline{L}assification via \underline{C}autious/\underline{A}ctive \underline{I}teration (or SEAL-C/AI in short). SEAL-C/AI adopt an iterative framework that takes turns to build or update two classifiers, one working at the graph instance level and the other at the hierarchical graph level. To simplify the representation of the hierarchical graph, we propose a novel supervised, self-attentive graph embedding method called SAGE, which embeds graph instances of arbitrary size into fixed-length vectors. Through experiments on synthetic data and Tencent QQ group data, we demonstrate that SEAL-C/AI not only outperform competing methods by a significant margin in terms of accuracy/Macro-F1, but also generate meaningful interpretations of the learned representations. </p>

This repository provides a PyTorch implementation of SEAL-CI as described in the paper:

> Semi-Supervised Graph Classification: A Hierarchical Graph Perspective.
> Zhang Xinyi, Lihui Chen.
> WWW, 2019.
> [[Paper]](https://arxiv.org/pdf/1904.05003.pdf)

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
torch-scatter     1.1.2
torch-sparse      0.2.2
torch-cluster     1.2.4
torch-geometric   1.0.3
torchvision       0.2.1
```
### Datasets
The code takes graphs for training from an input folder where each graph is stored as a JSON. Graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.

Every JSON file has the following key-value structure:

```javascript
{"edges": [[0, 1],[1, 2],[2, 3],[3, 4]],
 "labels": {"0": "A", "1": "B", "2": "C", "3": "A", "4": "B"},
 "target": 1}
```
The **edges** key has an edge list value which descibes the connectivity structure. The **labels** key has labels for each node which are stored as a dictionary -- within this nested dictionary labels are values, node identifiers are keys. The **target** key has an integer value which is the class membership.

### Outputs

The predictions are saved in the `output/` directory. Each embedding has a header and a column with the graph identifiers. Finally, the predictions are sorted by the identifier column.

### Options
Training a CapsGNN model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
  --prediction-path   STR    Output predictions file.     Default is `output/watts_predictions.csv`.
```
#### Model options
```
  --epochs                      INT     Number of epochs.                  Default is 10.
  --batch-size                  INT     Number fo graphs per batch.        Default is 8.
  --gcn-filters                 INT     Number of filters in GCNs.         Default is 2.
  --gcn-layers                  INT     Number of GCNs chained together.   Default is 5.
  --inner-attention-dimension   INT     Number of neurons in attention.    Default is 20.  
  --capsule-dimensions          INT     Number of capsule neurons.         Default is 8.
  --number-of-capsules          INT     Number of capsules in layer.       Default is 8.
  --weight-decay                FLOAT   Weight decay of Adam.              Defatuls is 10^-6.
  --lambd                       FLOAT   Regularization parameter.          Default is 1.0.
  --learning-rate               FLOAT   Adam learning rate.                Default is 0.01.
```
### Examples
The following commands learn a model and save the predictions. Training a model on the default dataset:
```
python src/main.py
```
<p align="center">
  <img width="500" src="seal.gif">
</p>

Training a CapsGNNN model for a 100 epochs.
```
python src/main.py --epochs 100
```
Changing the batch size.
```
python src/main.py --batch-size 128
```
