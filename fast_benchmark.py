"""
FAST RESEARCH BENCHMARK: Logic Manifold
========================================
Optimized for speed: Batch encoding, vectorized operations.
Target: N=20 samples per depth, full statistical reporting.
"""
import os
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 20
DEPTHS = [2, 3, 4]
BATCH_SIZE = 256
EPOCHS = 30  # Reduced for speed
LEARNING_RATE = 1e-4
MARGIN = 1.0
DATASET_SIZE = 10000  # Smaller for speed

# Variables
ALL_VARS = [f"V{i}" for i in range(50)]
VARS_TRAIN = ALL_VARS[:20]
VARS_TEST = ALL_VARS[30:]
OPS_TRAIN = ["AND", "OR"]

print(f"[INIT] Device: {DEVICE}")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# --------------------
# LOGIC ENGINE (Simplified)
# --------------------
def random_statement(depth, variables):
    if depth <= 1:
        return random.choice(variables)
    op = random.choice(OPS_TRAIN)
    left = random_statement(depth - 1, variables)
    right = random_statement(depth - 1, variables)
    return f"({left} {op} {right})"

def negate(stmt):
    if stmt.startswith("NOT "): return stmt[4:]
    return f"NOT {stmt}"

def generate_triplet(variables, depth):
    anchor = random_statement(depth, variables)
    positive = anchor  # Identity
    negative = negate(anchor)
    return anchor, positive, negative

# --------------------
# MODEL (Minimal SIREN)
# --------------------
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class FastManifold(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), Sine(30.0),
            nn.Linear(256, 256), Sine(1.0),
            nn.Linear(256, dim)
        )
    def forward(self, x):
        z = self.net(x)
        norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-7)
        return z / norm.clamp(min=1.0) * 0.99

def hyperbolic_dist(u, v, eps=1e-7):
    sq_dist = torch.sum((u - v) ** 2, dim=-1)
    u_norm_sq = torch.sum(u ** 2, dim=-1).clamp(max=1.0 - eps)
    v_norm_sq = torch.sum(v ** 2, dim=-1).clamp(max=1.0 - eps)
    num = 2 * sq_dist
    den = (1 - u_norm_sq) * (1 - v_norm_sq)
    arg = (1 + num / (den + eps)).clamp(min=1.0 + eps)
    return torch.acosh(arg)

# --------------------
# DATASET (Optimized)
# --------------------
class FastDataset(Dataset):
    def __init__(self, size, variables, depth):
        triplets = [generate_triplet(variables, depth) for _ in range(size)]
        all_txt = [t for trip in triplets for t in trip]
        print(f"[DATA] Encoding {len(all_txt)} samples...")
        all_emb = embedder.encode(all_txt, convert_to_numpy=True, show_progress_bar=False, batch_size=512)
        self.data = torch.tensor(all_emb, dtype=torch.float32).view(size, 3, -1)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --------------------
# TRAINING (Fast)
# --------------------
def train_fast():
    print("\n[TRAIN] Fast Training...")
    dataset = FastDataset(DATASET_SIZE, VARS_TRAIN, depth=2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = FastManifold(EMBED_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            anc, pos, neg = batch[:, 0], batch[:, 1], batch[:, 2]
            z_a, z_p, z_n = model(anc), model(pos), model(neg)
            d_p = hyperbolic_dist(z_a, z_p)
            d_n = hyperbolic_dist(z_a, z_n)
            loss = torch.relu(d_p - d_n + MARGIN).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
    return model

# --------------------
# BENCHMARK (Batched & Fast)
# --------------------
def run_fast_benchmark(model):
    print(f"\n[BENCH] Running N={N_SAMPLES} per depth...")
    results = []
    
    for d in DEPTHS:
        print(f"  -> Depth {d}...")
        
        # Pre-generate all statements
        anchors = [random_statement(d, VARS_TEST) for _ in range(N_SAMPLES)]
        negations = [negate(a) for a in anchors]
        
        # Batch encode ALL at once (FAST!)
        all_texts = anchors + negations
        all_embs = embedder.encode(all_texts, convert_to_numpy=True, show_progress_bar=False, batch_size=512)
        
        anc_embs = torch.tensor(all_embs[:N_SAMPLES], dtype=torch.float32).to(DEVICE)
        neg_embs = torch.tensor(all_embs[N_SAMPLES:], dtype=torch.float32).to(DEVICE)
        
        # RAW Transformer (Cosine Similarity)
        raw_sims = []
        for i in range(N_SAMPLES):
            cos = np.dot(all_embs[i], all_embs[N_SAMPLES + i])
            cos /= (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[N_SAMPLES + i]))
            raw_sims.append(cos)
        
        # Trained Manifold (Hyperbolic Distance)
        model.eval()
        with torch.no_grad():
            z_anc = model(anc_embs)
            z_neg = model(neg_embs)
            # Identity: anchor vs itself (should be 0 distance)
            z_anc_ident = model(anc_embs)
            d_ident = hyperbolic_dist(z_anc, z_anc_ident).cpu().numpy()
            d_neg = hyperbolic_dist(z_anc, z_neg).cpu().numpy()
        
        # Convert to similarity (higher = more similar)
        trained_ident_sims = np.exp(-d_ident)
        trained_neg_sims = np.exp(-d_neg)
        trained_gaps = trained_ident_sims - trained_neg_sims
        raw_gaps = 1.0 - np.array(raw_sims)  # Identity is 1.0, so gap = 1 - neg_sim
        
        results.append({
            "Depth": d,
            "N": N_SAMPLES,
            "Raw_NegSim_Mean": np.mean(raw_sims),
            "Raw_NegSim_Std": np.std(raw_sims),
            "Trained_NegSim_Mean": np.mean(trained_neg_sims),
            "Trained_NegSim_Std": np.std(trained_neg_sims),
            "Trained_Gap_Mean": np.mean(trained_gaps),
            "Trained_Gap_Std": np.std(trained_gaps)
        })
    
    return pd.DataFrame(results)

# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    model = train_fast()
    df = run_fast_benchmark(model)
    
    print("\n" + "="*70)
    print("RESEARCH REPORT: LOGIC MANIFOLD (N=20, Fast Mode)")
    print("="*70)
    print(df.to_string(index=False))
    
    # Key Metric
    avg_raw = df["Raw_NegSim_Mean"].mean()
    avg_trained = df["Trained_NegSim_Mean"].mean()
    print(f"\n[SUMMARY]")
    print(f"  Raw Transformer sees S vs NOT_S as: {avg_raw:.2%} similar")
    print(f"  Trained Manifold sees S vs NOT_S as: {avg_trained:.2%} similar")
    print(f"  Logical Differentiation Improvement: {(avg_raw - avg_trained) / avg_raw:.1%}")
    
    df.to_csv("fast_benchmark_results.csv", index=False)
    print("\n[SAVED] fast_benchmark_results.csv")
    print("="*70)
