# GRASP
# Dynamic Networks
# Link Prediction


import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from node2vec import Node2Vec
import random


# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)


# ================= LOAD CSV (PREVIOUSLY DRAWN COMBINATIONS) =================
csv_path = "/Users/milan/Desktop/GHQ/data/loto7h_4532_k100.csv"
df = pd.read_csv(csv_path)

NODES = list(range(1, 40))


# ================= CREATE SNAPSHOTS FROM CSV =================
snapshots = []
for _, row in df.iterrows():
    G = nx.Graph()
    G.add_nodes_from(NODES)
    nums = sorted(row.values.tolist())
    for u, v in combinations(nums, 2):
        G.add_edge(u, v)
    snapshots.append(G)


# ================= AGGREGATE GRAPH =================
G = nx.Graph()
G.add_nodes_from(NODES)
for g in snapshots:
    G.add_edges_from(g.edges())


# ================= CANDIDATE PAIRS (DETERMINISTIC) =================
pairs = [(u, v) for u in NODES for v in NODES if u < v]


# ================= LABELS FROM CSV =================
edge_set = set()
for g in snapshots:
    edge_set |= set(g.edges())

labels = np.array([1 if (u, v) in edge_set else 0 for u, v in pairs])


# ================= FEATURES =================
def features(G, pairs):
    from networkx.algorithms.link_prediction import (
        jaccard_coefficient,
        adamic_adar_index,
        preferential_attachment
    )

    cn = {(u,v): len(list(nx.common_neighbors(G,u,v))) for u,v in pairs}
    jc = {(u,v): p for u,v,p in jaccard_coefficient(G, pairs)}
    aa = {(u,v): p for u,v,p in adamic_adar_index(G, pairs)}
    pa = {(u,v): p for u,v,p in preferential_attachment(G, pairs)}

    X = np.array([
        [cn[(u,v)], jc[(u,v)], aa[(u,v)], pa[(u,v)]]
        for u,v in pairs
    ])
    return X

X = features(G, pairs)


# ================= STRUCTURAL MODEL =================
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X, labels)

"""
Computing transition probabilities: 100%|█| 39/39 [00:00<
Generating walks (CPU: 1): 100%|█| 50/50 [00:00<00:00, 94
"""


# ================= NODE2VEC EMBEDDING =================
n2v = Node2Vec(
    G,
    dimensions=32,
    walk_length=10,
    num_walks=50,
    workers=1,
    seed=SEED
)

model = n2v.fit(window=5, min_count=1)

emb = {int(n): model.wv[str(n)] for n in G.nodes()}

def edge_emb(u,v):
    return np.mean(np.concatenate([emb[u], emb[v]]))


# ================= SCORE ALL EDGES =================
edge_score = {}
for i,(u,v) in enumerate(pairs):
    s_struct = rf.predict(X[i].reshape(1,-1))[0]
    s_emb = edge_emb(u,v)
    edge_score[(u,v)] = s_struct + s_emb


# ================= DETERMINISTIC GRASP OPTIMIZATION =================
TOP_EDGES = 200

top_edges = sorted(
    edge_score.items(),
    key=lambda x: (-x[1], x[0])
)[:TOP_EDGES]

candidate_nodes = sorted(
    set([u for (u,v),_ in top_edges] + [v for (u,v),_ in top_edges])
)

best_combo = None
best_score = -1e18

for combo in combinations(candidate_nodes, 7):
    scores = []
    for u,v in combinations(combo,2):
        scores.append(edge_score.get((u,v), edge_score.get((v,u), 0)))

    score = float(np.mean(scores))

    if (
        score > best_score or
        (score == best_score and combo < best_combo)
    ):
        best_score = score
        best_combo = combo


# ================= RESULT =================
print()
print("PREDIKCIJA SLEDEĆE 7-ČLANE GRANE (CSV ceo):")
print(best_combo)
print("Skor:", best_score)
print()
"""
PREDIKCIJA SLEDEĆE 7-ČLANE GRANE (CSV ceo):
(4, 7, 19, 24, 26, 34, 37)
Skor: 0.9906696023064709
"""



"""
Seed je postavljen u:

1. Random module - random.seed(SEED)
2. NumPy - np.random.seed(SEED)
3. Train/test split - train_test_split 
   (..., random_state=SEED)
4. Node2Vec - Node2Vec (..., seed=SEED)
5. Bilo koji shuffle u kreiranju kandidata 
   ili negative pairs - koristi isti seed



Koristili smo CSV prethodno izvučenih 4532 kombinacija.
Kombinovali smo strukturni model (CN/JC/AA/PA) i Node2Vec embedding.
Implementirana je deterministička GRASP optimizacija za predikciju sledeće 7-člane grane.
Seed je fiksiran, tako da predikcija bude ista pri svakom pokretanju.
"""
