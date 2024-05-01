import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool
import numpy as np

##
## Get the number of nodes to use in the graph
##
## For this example, they'll just be a generated permutation of features
##
NUM_NODES: int = int(input("Enter the number of nodes: "))


##
## Define our Graph Neural Network
##
class GNN(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        ## Our Pointer Decoder (Linear layer to compute scores for each node)
        self.decoder = nn.Linear(hidden_dim, NUM_NODES)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        ## Aggregate node features to graph features
        x = global_max_pool(x, batch)

        ##
        ## Return the scores for each node (using our pointer decoder)
        ##
        ## We could use a softmax layer here to get a probability distribution
        ## but we'll just return the scores for now
        ##
        return self.decoder(x)


##
## Execute if this is the main file
##
if __name__ == "__main__":
    ## ## ## ## ## ## ## ##
    ##                   ##
    ## Init the data set ##
    ##                   ##
    ## ## ## ## ## ## ## ##

    ## Define node features
    nodes = [[0, 0], [1, 0], [0, 1], [1, 1]]  ## x,y coordinates of the nodes
    x = torch.tensor(nodes, dtype=torch.float)

    ##
    ## Define edges
    ##
    ## This basically just means we're connecting node at index 0 to the
    ## node at index 1, node at index 1 to the node at index 2, and so on...
    ##
    ## Notice that the edges are defined in a way that the graph is a cycle
    ## 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> ...
    ##
    connected_edges_indices = [[0, 1, 2, 3], [1, 2, 3, 0]]
    edge_index = torch.tensor(connected_edges_indices, dtype=torch.long)

    ##
    ## Define target sequence
    ##
    ## We'll define a random permutation of the node indices as the target sequence
    ##
    ord_perm = np.random.permutation(NUM_NODES).copy()[::1]
    target_sequence = torch.tensor(ord_perm, dtype=torch.float)

    ## Define data class
    data = Data(x=x, edge_index=edge_index)

    ## ## ## ## ## ## ##
    ##                ##
    ## Start training ##
    ##                ##
    ## ## ## ## ## ## ##

    model = GNN(node_features=2, hidden_dim=128, output_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    ## Epoch loop
    for epoch in range(100):
        optimizer.zero_grad()

        output = model(data.x, data.edge_index, data.batch)
        output = output.view(-1)

        loss = criterion(output, target_sequence)
        loss.backward()

        optimizer.step()

    ## ## ## ## ## ## ##
    ##                ##
    ## Test the model ##
    ##                ##
    ## ## ## ## ## ## ##

    output: torch.Tensor = model(data.x, data.edge_index, data.batch)
    output = output.view(-1)

    ## Convert the output to a regular array
    output = output.detach().numpy()

    ## Round the output to the nearest integer
    output = np.round(np.abs(output))

    ## Convert the target sequence to a regular array
    target = target_sequence.detach().numpy()

    ## Print the predicted sequence and the target sequence
    print(f"Predicted sequence: {output} | Target sequence: {target}")


## ## ## ## ## ##
##             ##
## Conclusion  ##
##             ##
## ## ## ## ## ##

## Input:
##  - x: our node features (positions x, y)
##  - edge_index: our edges (connected nodes) (these are the indices of the nodes)
##
## Model Input:
##  - Our Data which uses the x and edge_index
##  - Our target sequence of the shortest path of edges (these are the indices of the nodes)
##
## Model Output:
##  - The predicted sequence of the shortest path of edges (these are the indices of the nodes)
##
