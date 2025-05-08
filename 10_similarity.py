import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from torch.nn.functional import cosine_similarity

# === 设置路径 ===
embedding_path = 'Data/embedding/semantic_embedding.pt'
structure_path = 'Data/similarity/structure.txt'
output_dir = 'Data/similarity'
os.makedirs(output_dir, exist_ok=True)

# === 加载嵌入数据 ===
data = torch.load(embedding_path, map_location='cpu')
embedding_matrix = data['embedding_matrix']
index2id = data['index2id']
id2index = data['id2index']

# === 构建图结构 ===
G = nx.DiGraph()
with open(structure_path, 'r', encoding='utf-8') as f:
    for line in f:
        child, parent = line.strip().split()
        G.add_edge(child, parent)

# === 识别叶子节点 ===
leaves = [n for n in G.nodes if G.in_degree(n) == 0]

# === 图 A：同父类的叶子类之间的相似度 ===
similarities_A = []
for parent in G.nodes:
    children = list(G.predecessors(parent))  # 子类 → 父类
    leaf_children = [c for c in children if c in leaves and c in id2index]
    for c1, c2 in combinations(leaf_children, 2):
        i1, i2 = id2index[c1], id2index[c2]
        v1, v2 = embedding_matrix[i1], embedding_matrix[i2]
        sim = cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        similarities_A.append(sim)

# === 图 B：每个叶子类与其直接父类的相似度 ===
similarities_B = []
for leaf in leaves:
    if leaf in id2index:
        parents = list(G.successors(leaf))  # 子 → 父
        for parent in parents:
            if parent in id2index:
                i1, i2 = id2index[leaf], id2index[parent]
                v1, v2 = embedding_matrix[i1], embedding_matrix[i2]
                sim = cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                similarities_B.append(sim)

# === 绘图 ===
sns.set(style="whitegrid")

# 图 A：兄弟叶子类对相似度
plt.figure(figsize=(8, 5))
sns.kdeplot(similarities_A, fill=True, color='skyblue')
plt.title("KDE of Similarity Between Sibling Leaf Nodes")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'leaf_leaf_similarity.png'))
plt.close()

# 图 B：叶子类与其父类相似度
plt.figure(figsize=(8, 5))
sns.kdeplot(similarities_B, fill=True, color='salmon')
plt.title("KDE of Similarity Between Leaf Nodes and Their Parents")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'leaf_parent_similarity.png'))
plt.close()
