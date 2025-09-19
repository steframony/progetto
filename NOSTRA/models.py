# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def pad2batch(pos):
    """
    Converte un tensore "paddato" di indici di nodi in una lista piatta
    di indici di nodi e un vettore di assegnazione al batch corrispondente.
    
    Args:
        pos: Un tensore di forma [numero_sottografi, max_nodi_per_sottografo]
             riempito con -1 per il padding.
    
    Returns:
        Una tupla contenente:
        - batch (Tensor): Un vettore che mappa ogni nodo al suo indice di sottografo.
        - flat_pos (Tensor): Un tensore piatto di indici di nodi validi.
    """
    # Crea una maschera per gli indici di nodi validi (diversi da -1)
    mask = (pos != -1)
    # Calcola il numero di nodi validi in ogni sottografo
    num_nodes_per_subgraph = mask.sum(dim=1)
    # Crea il vettore batch ripetendo l'indice di ogni sottografo
    batch = torch.arange(pos.shape[0], device=pos.device).repeat_interleave(num_nodes_per_subgraph)
    # Appiattisce il tensore di posizioni per ottenere una lista 1D di indici di nodi validi
    flat_pos = pos[mask]
    return batch, flat_pos


class SimpleGNN(nn.Module):
    """
    Una semplice architettura Graph Neural Network per la classificazione di sottografi.
    
    Questo modello è composto da un layer di embedding per le feature dei nodi, una serie
    di layer GCNConv per il message passing, un layer di global mean pooling per
    aggregare gli embedding dei nodi in una rappresentazione del sottografo, e infine
    un MLP per la classificazione.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(SimpleGNN, self).__init__()
        
        # Layer di embedding per feature intere dei nodi (es. grado)
        self.input_emb = nn.Embedding(input_dim + 1, hidden_dim)

        self.convs = nn.ModuleList()
        # Primo layer GCN
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # Layer GCN successivi
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Tasso di dropout per la regolarizzazione
        self.dropout = dropout

        # Classificatore MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, subg_nodes, z=None, id=0):
        """
        Forward pass per SimpleGNN.
        
        Args:
            x (Tensor): Feature dei nodi per l'intero grafo.
            edge_index (Tensor): Indici degli archi per l'intero grafo.
            edge_attr (Tensor): Attributi degli archi (non usati in questo modello).
            subg_nodes (Tensor): Tensore paddato di indici di nodi per ogni sottografo nel batch.
            z (Tensor, optional): Label dei nodi per ZGDataloader. Default a None.
            id (int, optional): ID per multi-task learning (non usato). Default a 0.

        Returns:
            Tensor: I logit di output per ogni sottografo nel batch.
        """
        # 1. Ottiene gli embedding iniziali dei nodi per l'intero grafo.
        # Ci si aspetta che 'x' contenga valori interi (come i gradi).
        # .squeeze() rimuove le dimensioni singole.
        x_emb = self.input_emb(x.squeeze())
        
        # 2. Applica i layer GCN con attivazione ReLU e dropout.
        for conv in self.convs:
            x_emb = conv(x_emb, edge_index)
            x_emb = F.relu(x_emb)
            x_emb = F.dropout(x_emb, p=self.dropout, training=self.training)
            
        # 3. Effettua il pooling degli embedding dei nodi per ottenere gli embedding dei sottografi.
        # `pad2batch` converte il tensore `subg_nodes` in un formato
        # adatto per le funzioni di global pooling di PyG.
        batch, pos = pad2batch(subg_nodes)
        
        # Seleziona gli embedding per i nodi che fanno parte dei sottografi in questo batch.
        subgraph_node_features = x_emb[pos]
        
        # Applica il global mean pooling per ottenere un singolo embedding per ogni sottografo.
        subgraph_embedding = global_mean_pool(subgraph_node_features, batch)
        
        # 4. Passa gli embedding dei sottografi attraverso il classificatore MLP finale.
        out = self.mlp(subgraph_embedding)
        
        return out