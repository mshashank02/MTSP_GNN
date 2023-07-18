import torch

def edge_index_representation(num_nodes_type1, num_nodes_type2):
    # Calculate the number of edges (bidirectional) between different types of nodes
    num_edges = num_nodes_type1 * num_nodes_type2

    # Create arrays to store the source and target node indices
    source_indices = torch.zeros(num_edges , dtype=torch.long)
    target_indices = torch.zeros(num_edges , dtype=torch.long)

    # Fill the arrays with the source and target indices for all edges
    idx = 0
    for i in range(num_nodes_type1):
        for j in range(num_nodes_type2):
            
            source_indices[idx] = j + num_nodes_type1  # Bidirectional edge
            target_indices[idx] = i
            idx += 1

    # Create the edge index tensor
    edge_index = torch.stack([source_indices, target_indices], dim=0)

    return edge_index

# Example usage with 3 nodes of type 1 and 2 nodes of type 2
num_nodes_type1 = 2
num_nodes_type2 = 2
edge_index = edge_index_representation(num_nodes_type1, num_nodes_type2)
print(edge_index)
