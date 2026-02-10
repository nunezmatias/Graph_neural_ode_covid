#%% MAKE GNN-COMPATIBLE GRAPH

function build_GNNGraph(adj)
    # rearrange attributes for GNN
    ordered_matrices = [features[k] for k in states]
    attributes = stack(ordered_matrices)
    return GNNGraph(adj, ndata=attributes)
end