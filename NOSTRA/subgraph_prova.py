import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, to_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# =========================
# BaseGraph Definition
# =========================
class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_label, mask):
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.mask = mask
        self.to_undirected()

    def addOneFeature(self):
        self.x = torch.cat(
            (self.x, torch.ones(self.x.shape[0], self.x.shape[1], 1)),
            dim=-1)

    def setOneFeature(self):
        self.x = torch.ones((self.x.shape[0], 1, 1), dtype=torch.int64)

    def get_split(self, split: str):
        tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        return self.x, self.edge_index, self.edge_attr, self.pos[self.mask == tar_mask], self.y[self.mask == tar_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(self.edge_index, self.edge_attr)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        return self


# =========================
# Dataset Loader
# =========================
def load_dataset(name: str):
    if name in ["coreness", "cut_ratio", "density", "component"]:
        obj = np.load(f"./progetto/dataset_/{name}/tmp.npy", allow_pickle=True).item()
        edge = np.array([[i[0] for i in obj['G'].edges],
                         [i[1] for i in obj['G'].edges]])
        node = [n for n in obj['G'].nodes]
        subG = obj["subG"]
        subG_pad = pad_sequence([torch.tensor(i) for i in subG],
                                batch_first=True,
                                padding_value=-1)
        subGLabel = torch.tensor([ord(i) - ord('A') for i in obj["subGLabel"]])
        cnt = subG_pad.shape[0]
        mask = torch.cat(
            (torch.zeros(cnt - cnt // 2, dtype=torch.int64),
             torch.ones(cnt // 4, dtype=torch.int64),
             2 * torch.ones(cnt // 2 - cnt // 4, dtype=torch.int64)))
        mask = mask[torch.randperm(mask.shape[0])]
        return BaseGraph(torch.empty((len(node), 1, 0)),
                         torch.from_numpy(edge),
                         torch.ones(edge.shape[1]),
                         subG_pad,
                         subGLabel,
                         mask)
    else:
        raise NotImplementedError()


# =========================
# GNN Model
# =========================
class GNNSubgraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, subgraphs):
        x = self.conv1(x.squeeze(1), edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        subgraph_embeddings = []
        for subgraph in subgraphs:
            valid_nodes = subgraph[subgraph >= 0]
            sg_emb = x[valid_nodes].mean(dim=0)
            subgraph_embeddings.append(sg_emb)

        subgraph_embeddings = torch.stack(subgraph_embeddings)
        out = self.classifier(subgraph_embeddings)
        return out


# =========================
# Training Function
# =========================
def train(model, data, optimizer, criterion, device):
    model.train()
    x, edge_index, edge_attr, pos, y = data.get_split("train")
    x, edge_index, pos, y = x.to(device), edge_index.to(device), pos.to(device), y.to(device)

    optimizer.zero_grad()
    out = model(x, edge_index, pos)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


# =========================
# Test Function
# =========================
def test(model, data, device, split="test"):
    model.eval()
    x, edge_index, edge_attr, pos, y = data.get_split(split)
    x, edge_index, pos, y = x.to(device), edge_index.to(device), pos.to(device), y.to(device)

    with torch.no_grad():
        out = model(x, edge_index, pos)
        pred = out.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return acc


# =========================
# Visualization
# =========================
def visualize_subgraph(data, subgraph_id):
    sub_nodes = data.pos[subgraph_id]
    valid_nodes = sub_nodes[sub_nodes >= 0].tolist()

    G = to_networkx(data, to_undirected=True)
    sg = G.subgraph(valid_nodes)

    plt.figure(figsize=(6, 6))
    nx.draw(sg, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)
    plt.title(f"Subgraph ID: {subgraph_id}")
    plt.show()


# =========================
# Main Function
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica dataset
    dataset_name = "ppi_bp" 
    data = load_dataset(dataset_name).to(device)
    data.setOneFeature()  # imposta feature costanti per i nodi

    in_channels = data.x.shape[-1]
    hidden_channels = 64
    num_classes = len(data.y.unique())

    model = GNNSubgraphClassifier(in_channels, hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Addestramento
    for epoch in range(1, 51):
        loss = train(model, data, optimizer, criterion, device)
        acc = test(model, data, device, split="test")
        print(f"[Epoch {epoch:02d}] Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    # Visualizzazione sottografo
    visualize_subgraph(data, subgraph_id=0)


if __name__ == "__main__":
    main()
