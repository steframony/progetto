from models import SimpleGNN
from SubGDataset import GDataset, GDataloader, ZGDataloader
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import numpy as np
import time
import random
import train
import metrics

# --- Argomenti da riga di comando ---
parser = argparse.ArgumentParser(description='Test SimpleGNN for Subgraph Classification')
parser.add_argument('--dataset', type=str, default='ppi_bp')
parser.add_argument('--use_deg', action='store_true', default=True)
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
parser.add_argument('--use_maxzeroone', action='store_true')
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
args = parser.parse_args()

# --- Setup del dispositivo ---
device = f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu'

def set_seed(seed: int):
    print(f"Impostazione seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Variabili globali ---
trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn, loss_fn = None, None
loader_fn = GDataloader
tloader_fn = GDataloader

# --- Caricamento e divisione dei dati ---
def split():
    global trn_dataset, val_dataset, tst_dataset, baseG, max_deg, output_channels, loader_fn, tloader_fn, loss_fn, score_fn
    
    baseG = datasets.load_dataset(args.dataset)
    
    if baseG.y.unique().shape[0] == 2:
        loss_fn = lambda x, y: BCEWithLogitsLoss()(x.view(-1), y.view(-1))
        baseG.y = baseG.y.to(torch.float)
        output_channels = 1 if baseG.y.ndim == 1 else baseG.y.shape[1]
        score_fn = metrics.binaryf1
    else:
        loss_fn = CrossEntropyLoss()
        baseG.y = baseG.y.to(torch.int64)
        output_channels = baseG.y.unique().shape[0]
        score_fn = metrics.microf1

    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        print("Nessuna feature specificata. Utilizzo 'use_deg' di default.")
        baseG.setDegreeFeature()

    max_deg = int(torch.max(baseG.x))
    baseG.to(device)
    
    trn_dataset = GDataset(*baseG.get_split("train"))
    val_dataset = GDataset(*baseG.get_split("valid"))
    tst_dataset = GDataset(*baseG.get_split("test"))
    
    global loader_fn, tloader_fn
    if args.use_maxzeroone:
        z_fn = lambda x, y: torch.zeros((x.shape[0], x.shape[1]), dtype=torch.int64) # Placeholder
        loader_fn = lambda ds, bs: ZGDataloader(ds, bs, z_fn=z_fn)
        tloader_fn = lambda ds, bs: ZGDataloader(ds, bs, z_fn=z_fn, shuffle=True, drop_last=False)
    else:
        loader_fn = lambda ds, bs: GDataloader(ds, bs)
        tloader_fn = lambda ds, bs: GDataloader(ds, bs, shuffle=True)

# --- Costruzione del modello ---
def build_simple_model(hidden_dim, conv_layer, dropout):
    model = SimpleGNN(
        input_dim=max_deg,
        hidden_dim=hidden_dim,
        output_dim=output_channels,
        num_layers=conv_layer,
        dropout=dropout
    ).to(device)
    return model

# --- Funzione di Test Principale ---
def test(hidden_dim=64, conv_layer=3, dropout=0.5, lr=1e-3, batch_size=128, resi=0.5):
    num_div = tst_dataset.y.shape[0] / batch_size
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f"--- Ripetizione {repeat} ---")
        split()
        
        gnn = build_simple_model(hidden_dim, conv_layer, dropout)
        
        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)
        
        optimizer = Adam(gnn.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=1e-5, patience=10)

        best_val_score = 0
        final_tst_score = 0
        early_stop_counter = 0
        
        for epoch in range(1, 301):
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            
            if epoch > 10:
                val_score, val_loss = train.test(gnn, val_loader, score_fn, loss_fn=loss_fn)
                scheduler.step(val_loss)
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    final_tst_score, _ = train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)
                    early_stop_counter = 0
                    print(f"Epoch {epoch}: Loss {loss:.4f}, Val Score {val_score:.4f}, Test Score {final_tst_score:.4f}")
                else:
                    early_stop_counter += 1
            
            if early_stop_counter > 25: # Pazienza di 25 epoche
                print("Early stopping.")
                break
        
        print(f"Fine Ripetizione {repeat}: Miglior Val {best_val_score:.3f}, Test Finale {final_tst_score:.3f}")
        outs.append(final_tst_score)
        
    print("\n--- Risultati Finali ---")
    if len(outs) > 1:
        print(f"Score Medio su Test: {np.average(outs):.3f} ± {np.std(outs):.3f}")
    else:
        print(f"Score Finale su Test: {outs[0]:.3f}")

# --- Esecuzione ---
if __name__ == '__main__':
    print("Argomenti:", args)

    # Definisci qui gli iperparametri per il modello SimpleGNN
    params = {
        'hidden_dim': 64,
        'conv_layer': 3,
        'dropout': 0.5,
        'lr': 0.001,
        'batch_size': 128,
        'resi': 0.5 # Fattore per lo scheduler del learning rate
    }

    print("Parametri utilizzati:", params)
    split() # Chiamata iniziale per inizializzare i dataset
    test(**params)