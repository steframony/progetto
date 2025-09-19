import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import networkx as nx
from dataset import load_dataset


# ======== MODEL ==========
class SubgraphGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SubgraphGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out


# ======== UTILS ==========
def get_subgraph_node_features(graph, subgraphs):
    all_nodes = []
    batch = []
    for i, nodes in enumerate(subgraphs):
        nodes = nodes[nodes != -1]
        all_nodes.extend(nodes.tolist())
        batch.extend([i] * len(nodes))
    x = graph.x.squeeze(1)
    node_features = x[all_nodes]
    batch = torch.tensor(batch, dtype=torch.long)
    return node_features, batch, all_nodes


def train(model, graph, optimizer, split='train'):
    model.train()
    optimizer.zero_grad()
    x, edge_index, edge_attr, subgraphs, labels = graph.get_split(split)

    node_features, batch, all_nodes = get_subgraph_node_features(graph, subgraphs)
    all_nodes_tensor = torch.tensor(all_nodes, dtype=torch.long)
    edge_mask = torch.isin(graph.edge_index[0], all_nodes_tensor) & torch.isin(graph.edge_index[1], all_nodes_tensor)
    edge_index = graph.edge_index[:, edge_mask]

    out = model(node_features.float(), edge_index, batch)
    loss = F.cross_entropy(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, graph, split='test'):
    model.eval()
    x, edge_index, edge_attr, subgraphs, labels = graph.get_split(split)

    node_features, batch, all_nodes = get_subgraph_node_features(graph, subgraphs)
    all_nodes_tensor = torch.tensor(all_nodes, dtype=torch.long)
    edge_mask = torch.isin(graph.edge_index[0], all_nodes_tensor) & torch.isin(graph.edge_index[1], all_nodes_tensor)
    edge_index = graph.edge_index[:, edge_mask]

    with torch.no_grad():
        out = model(node_features.float(), edge_index, batch)
        pred = out.argmax(dim=1)
        acc = (pred == labels).float().mean().item()
    return acc


# ======== VISUALIZATION ==========
def plot_subgraph(graph, subgraph_nodes, labels=None, pred_labels=None):
    sub_nodes = subgraph_nodes[subgraph_nodes != -1].tolist()
    G = nx.Graph()
    edge_index = graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if u in sub_nodes and v in sub_nodes:
            G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, edge_color='gray')
    title = "Subgraph"
    if labels is not None:
        title += f" | Label: {labels}"
    if pred_labels is not None:
        title += f" | Pred: {pred_labels}"
    plt.title(title)
    plt.show()


def visualize_example(graph, model):
    model.eval()
    x, edge_index, edge_attr, subgraphs, labels = graph.get_split("test")
    node_features, batch, all_nodes = get_subgraph_node_features(graph, subgraphs)
    all_nodes_tensor = torch.tensor(all_nodes, dtype=torch.long)
    edge_mask = torch.isin(graph.edge_index[0], all_nodes_tensor) & torch.isin(graph.edge_index[1], all_nodes_tensor)
    edge_index = graph.edge_index[:, edge_mask]

    with torch.no_grad():
        out = model(node_features.float(), edge_index, batch)
        pred = out.argmax(dim=1)

    idx = torch.randint(0, subgraphs.shape[0], (1,)).item()
    subgraph_nodes = subgraphs[idx]
    label = labels[idx].item()
    pred_label = pred[idx].item()
    plot_subgraph(graph, subgraph_nodes, label, pred_label)


# ======== MAIN ==========
def main():
    dataset_name = "ppi_bp"  # Cambia con "density", "ppi_bp", ecc.
    graph = load_dataset(dataset_name)
    graph.setOneFeature()  # Scegli una: setOneFeature(), setDegreeFeature(), ecc.

    input_dim = graph.x.shape[-1]
    hidden_dim = 64
    output_dim = int(graph.y.max().item()) + 1

    model = SubgraphGNN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        loss = train(model, graph, optimizer)
        acc = test(model, graph)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    # Visualizza un sottografo del test set
    visualize_example(graph, model)


if __name__ == "__main__":
    main()
