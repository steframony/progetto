import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from dataset import load_dataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
import torch_scatter

class SubgraphGNN(nn.Module):
    """
    Modello GNN per la classificazione di sottografi.
    
    Il modello funziona in tre fasi principali:
    1. Embedding dei nodi tramite GNN layers
    2. Aggregazione delle rappresentazioni dei nodi per ogni sottografo
    3. Classificazione finale dei sottografi
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, 
                 gnn_type='GCN', pooling='mean', dropout=0.5):
        """
        Args:
            input_dim: Dimensione delle features iniziali dei nodi
            hidden_dim: Dimensione hidden del GNN
            num_classes: Numero di classi per la classificazione
            num_layers: Numero di layer GNN
            gnn_type: Tipo di GNN ('GCN' o 'GAT')
            pooling: Tipo di pooling ('mean', 'max', 'add', 'attention')
            dropout: Probabilità di dropout
        """
        super(SubgraphGNN, self).__init__()
        
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.dropout = dropout
        
        # Layer GNN per l'embedding dei nodi
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'GCN':
            # Graph Convolutional Network layers
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'GAT':
            # Graph Attention Network layers
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
                else:
                    self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Attention pooling se specificato
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        """
        Args:
            data: BaseGraph object contenente:
                - x: node features [num_nodes, num_features, feature_dim]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: edge weights [num_edges]
                - pos: subgraph nodes [num_subgraphs, max_subgraph_size] (-1 per padding)
        
        Returns:
            logits: [num_subgraphs, num_classes]
        """
        x, edge_index, edge_attr, subgraph_nodes = data.x, data.edge_index, data.edge_attr, data.pos
        
        x = x.float()
        # Reshape node features se necessario
        if len(x.shape) == 3:
            x = x.squeeze(1)  # [num_nodes, feature_dim]
        
        # Fase 1: Embedding dei nodi tramite GNN
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GCN':
                x = gnn_layer(x, edge_index, edge_attr)
            else:  # GAT
                x = gnn_layer(x, edge_index)
            
            # Applicazione batch normalization e attivazione
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            # Dropout tranne nell'ultimo layer
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Fase 2: Aggregazione per ogni sottografo
        subgraph_embeddings = []
        
        for subgraph_idx in range(subgraph_nodes.size(0)):
            # Ottieni i nodi del sottografo (escludi padding -1)
            nodes = subgraph_nodes[subgraph_idx]
            valid_nodes = nodes[nodes != -1]
            
            if len(valid_nodes) == 0:
                # Sottografo vuoto, usa embedding zero
                subgraph_emb = torch.zeros(x.size(1), device=x.device)
            else:
                # Estrai le embeddings dei nodi del sottografo
                node_embeddings = x[valid_nodes]
                
                # Applica pooling per ottenere rappresentazione del sottografo
                if self.pooling == 'mean':
                    subgraph_emb = torch.mean(node_embeddings, dim=0)
                elif self.pooling == 'max':
                    subgraph_emb = torch.max(node_embeddings, dim=0)[0]
                elif self.pooling == 'add':
                    subgraph_emb = torch.sum(node_embeddings, dim=0)
                elif self.pooling == 'attention':
                    # Attention-based pooling
                    attention_weights = self.attention(node_embeddings)
                    attention_weights = F.softmax(attention_weights, dim=0)
                    subgraph_emb = torch.sum(attention_weights * node_embeddings, dim=0)
            
            subgraph_embeddings.append(subgraph_emb)
        
        # Stack delle embeddings dei sottografi
        subgraph_embeddings = torch.stack(subgraph_embeddings)
        
        # Fase 3: Classificazione finale
        logits = self.classifier(subgraph_embeddings)
        
        return logits


class SubgraphGNNWithReadout(nn.Module):
    """
    Versione alternativa che usa readout functions più sofisticate
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, 
                 gnn_type='GCN', dropout=0.5):
        super(SubgraphGNNWithReadout, self).__init__()
        
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        if gnn_type == 'GCN':
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
                else:
                    self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'GAT':
            for i in range(num_layers):
                if i == 0:
                    self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
                else:
                    self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Multi-scale readout: combina mean, max e sum pooling
        self.readout_dim = hidden_dim * 3  # mean + max + sum
        
        # Classificatore
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr, subgraph_nodes = data.x, data.edge_index, data.edge_attr, data.pos
        
        x = x.float()

        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        # GNN forward pass
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GCN':
                x = gnn_layer(x, edge_index, edge_attr)
            else:
                x = gnn_layer(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Multi-scale readout per ogni sottografo
        subgraph_embeddings = []
        
        for subgraph_idx in range(subgraph_nodes.size(0)):
            nodes = subgraph_nodes[subgraph_idx]
            valid_nodes = nodes[nodes != -1]
            
            if len(valid_nodes) == 0:
                subgraph_emb = torch.zeros(self.readout_dim, device=x.device)
            else:
                node_embeddings = x[valid_nodes]
                
                # Combina diversi tipi di pooling
                mean_pool = torch.mean(node_embeddings, dim=0)
                max_pool = torch.max(node_embeddings, dim=0)[0]
                sum_pool = torch.sum(node_embeddings, dim=0)
                
                # Concatena le rappresentazioni
                subgraph_emb = torch.cat([mean_pool, max_pool, sum_pool], dim=0)
            
            subgraph_embeddings.append(subgraph_emb)
        
        subgraph_embeddings = torch.stack(subgraph_embeddings)
        logits = self.classifier(subgraph_embeddings)
        
        return logits


# Funzione di training
def train_model(model, data, optimizer, criterion, mask_type='train'):
    """
    Funzione per il training del modello
    
    Args:
        model: Il modello GNN
        data: BaseGraph object
        optimizer: Ottimizzatore
        criterion: Loss function
        mask_type: 'train', 'valid', o 'test'
    
    Returns:
        loss: Loss value
        accuracy: Accuracy
    """
    model.train()
    
    # Ottieni i dati per il set specificato
    x, edge_index, edge_attr, subgraph_nodes, labels = data.get_split(mask_type)
    
    # Crea un oggetto data per il forward pass
    batch_data = type('Data', (), {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'pos': subgraph_nodes
    })()
    
    optimizer.zero_grad()
    logits = model(batch_data)
    
    # Calcola loss e accuracy
    if len(labels.shape) > 1:  # Multi-label classification
        loss = criterion(logits, labels)
        predicted = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (predicted == labels).float().mean()
    else:  # Single-label classification
        loss = criterion(logits, labels.long())
        predicted = torch.argmax(logits, dim=1)
        accuracy = (predicted == labels).float().mean()
    
    if mask_type == 'train':
        loss.backward()
        optimizer.step()
    
    return loss.item(), accuracy.item()


def main():
    """
    Main function per training e testing del modello SubgraphGNN
    """
    # Carica il dataset
    dataset_name = "ppi_bp"  # Cambia con "density", "ppi_bp", "coreness", ecc.
    print(f"Caricando dataset: {dataset_name}")
    
    try:
        graph = load_dataset(dataset_name)
        print(f"Dataset caricato con successo!")
        print(f"Numero di nodi: {graph.x.shape[0] if hasattr(graph, 'x') else 'N/A'}")
    except Exception as e:
        print(f"Errore nel caricamento del dataset: {e}")
        return
    
    # Configura le features dei nodi
    # Prova diverse configurazioni:
    # graph.setOneFeature()      # Feature costanti
    # graph.setDegreeFeature()   # Feature basate sul degree
    # graph.setRandomFeature()   # Feature random
    
    # Per questo esempio, proviamo a settare le feature
    try:
        if hasattr(graph, 'setDegreeFeature'):
            graph.setDegreeFeature()
            print("Feature degree impostate")
        else:
            print("Metodo setDegreeFeature non disponibile, usando feature esistenti")
    except Exception as e:
        print(f"Errore nell'impostazione delle feature: {e}")
        return
    
    # Parametri del modello
    input_dim = graph.x.shape[-1]
    hidden_dim = 64
    
    # Determina il numero di classi
    if hasattr(graph, 'y') and graph.y is not None:
        if len(graph.y.shape) > 1:  # Multi-label
            num_classes = graph.y.shape[1]
            is_multilabel = True
        else:  # Single-label
            num_classes = int(graph.y.max().item()) + 1
            is_multilabel = False
    else:
        print("Errore: il dataset non ha labels (y)")
        return
    
    print(f"Dimensioni:")
    print(f"  Input: {input_dim}")
    print(f"  Hidden: {hidden_dim}")
    print(f"  Output: {num_classes}")
    print(f"  Multi-label: {is_multilabel}")
    
    # Crea il modello
    model = SubgraphGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=3,
        gnn_type='GCN',  # Prova anche 'GAT'
        pooling='attention',  # Prova 'mean', 'max', 'add'
        dropout=0.5
    )
    
    print(f"Modello creato con {sum(p.numel() for p in model.parameters())} parametri")
    
    # Ottimizzatore e loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    if is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nInizio training...")
    best_val_acc = 0.0
    
    for epoch in range(1, 101):
        # Training
        train_loss, train_acc = train_model(model, graph, optimizer, criterion, 'train')
        
        # Validation (se disponibile)
        try:
            val_loss, val_acc = train_model(model, graph, optimizer, criterion, 'valid')
            
            # Test
            test_loss, test_acc = train_model(model, graph, optimizer, criterion, 'test')
            
            # Stampa risultati ogni 10 epoche
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
            
            # Salva il miglior modello
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_{dataset_name}.pth')
        
        except Exception as e:
            # Se non ci sono split train/val/test, usa solo training
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    
    print(f"\nTraining completato! Miglior validation accuracy: {best_val_acc:.4f}")
    
    # Carica il miglior modello per il test finale
    try:
        model.load_state_dict(torch.load(f'best_model_{dataset_name}.pth'))
        final_test_loss, final_test_acc = train_model(model, graph, optimizer, criterion, 'test')
        print(f"Test finale con miglior modello: Accuracy {final_test_acc:.4f}")
    except:
        print("Test finale non disponibile")


if __name__ == "__main__":
    main()
