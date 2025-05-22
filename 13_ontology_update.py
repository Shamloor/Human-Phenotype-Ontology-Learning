import torch
import networkx as nx
import pandas as pd
from torch.nn.functional import cosine_similarity

# === 第一步：构建图结构 ===
structure_path = 'Data/embedding/structure.txt'
G = nx.DiGraph()

def is_valid_node(node):
    return node.startswith("<HP_") or node == "<Thing>"

with open(structure_path, 'r', encoding='utf-8') as f:
    for line in f:
        child, parent = line.strip().split()
        if is_valid_node(child) and is_valid_node(parent):
            G.add_edge(child, parent)

# === 第二步：计算相似度 ===
new_path = 'Data/embedding/new_items_embedding.pt'
ontology_path = 'Data/embedding/ontology_embedding.pt'

new_data = torch.load(new_path, map_location='cpu')
onto_data = torch.load(ontology_path, map_location='cpu')

new_matrix = new_data['embedding_matrix']
onto_matrix = onto_data['embedding_matrix']
onto_index2id = onto_data['index2id']
onto_id2index = onto_data['id2index']
new_index2id = new_data['index2id']

matches = []

for i, new_vec in enumerate(new_matrix):
    sims = cosine_similarity(new_vec.unsqueeze(0), onto_matrix).squeeze()
    sorted_indices = torch.argsort(sims, descending=True)  # 相似度从高到低排序

    for idx in sorted_indices:
        idx = idx.item()
        matched_node = onto_index2id[idx]
        if is_valid_node(matched_node):
            best_score = sims[idx].item()
            matches.append([new_index2id[i], matched_node, best_score])
            break  # 一旦找到合法匹配就停止
    else:
        print(f"[Skip] No valid match found for {new_index2id[i]}")

# === 第三步：根据相似度修改图结构 ===
for new_id, target_node, score in matches:
    new_node = f"{new_id}"
    if score < 0.60:
        # 插入为兄弟结点
        parents = list(G.successors(target_node))
        for p in parents:
            G.add_edge(new_node, p)
    elif score < 0.81:
        # 插入为子类
        G.add_edge(new_node, target_node)
    else:
        # 重命名为现有结点 + new_id
        renamed = f"{target_node}/{new_id}"
        G = nx.relabel_nodes(G, {target_node: renamed})
        # 更新 target_node 也要变了，用于后续最近祖先查找
        for m in matches:
            if m[1] == target_node:
                m[1] = renamed

# === 第四步：查找最近共同祖先并保存 ===
map_path = "Data/classes/classes_pdf_map.txt"
uri_to_pmid = {}

def get_all_ancestors(G, node):
    ancestors = set()
    frontier = [node]
    while frontier:
        current = frontier.pop()
        for parent in G.successors(current):  # 向上遍历
            if parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)
    return ancestors

with open(map_path, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            uri, pmid = parts
            node_id = "<" + uri.split('/')[-1] + ">"
            pmid = pmid.replace("PMID:", "")
            uri_to_pmid[pmid] = node_id

results = []
depth_bins = [0, 1, 2, 3, 4]
depth_counts = {k: 0 for k in depth_bins}
depth_counts['5+'] = 0

for new_id, matched_node, _ in matches:
    pmid = new_id
    if pmid not in uri_to_pmid:
        continue
    original_node = uri_to_pmid[pmid]
    new_node = f"{pmid}" if matched_node.find(f"/{pmid}") == -1 else matched_node

    if original_node not in G or new_node not in G:
        continue

    # 找共同祖先
    try:
        ancestors_1 = get_all_ancestors(G, original_node) | {original_node}
        ancestors_2 = get_all_ancestors(G, new_node) | {new_node}
        common = ancestors_1 & ancestors_2

        if not common:
            print("allert")
            continue

        # 找距离最近的共同祖先
        best_common = None
        best_score = float('inf')
        for c in common:
            try:
                d1 = nx.shortest_path_length(G, source=original_node, target=c)
                d2 = nx.shortest_path_length(G, source=new_node, target=c)
                avg_d = (d1 + d2) / 2
                if avg_d < best_score:
                    best_score = avg_d
                    best_common = c
            except:
                continue

        if best_common:
            results.append([original_node, pmid, best_common, round(best_score, 2)])
            bin_label = int(best_score) if best_score < 5 else '5+'
            depth_counts[bin_label] += 1
    except:
        continue

# 保存为CSV
df_result = pd.DataFrame(results, columns=["original", "pmid", "common_ancestor", "avg_distance"])
output_path = "Data/evaluation/common_ancestor.csv"
df_result.to_csv(output_path, index=False)

# === 第五步：绘制柱状图 ===
import matplotlib.pyplot as plt

labels = ['0', '1', '2', '3', '4', '5+']
values = []
for k in labels:
    if k.isdigit():
        values.append(depth_counts.get(int(k), 0))
    else:
        values.append(depth_counts.get(k, 0))

plt.figure(figsize=(6.5, 4.5))
bars = plt.bar(labels, values, color='lightgray', edgecolor='black', linewidth=1.2)
plt.xlabel("Average Distance to Common Ancestor", labelpad=6)
plt.ylabel("Number of Cases", labelpad=6)
plt.title("Distribution of Ontological Proximity", pad=10)

# 留出图像顶部空间
plt.ylim(0, max(values) * 1.15)

# 添加数值标签，避免与边框重叠
for bar in bars:
    height = bar.get_height()
    offset = max(1, max(values) * 0.03)  # 最少偏移1，高度越大偏得越高
    plt.text(bar.get_x() + bar.get_width()/2, height + offset, f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout(pad=0.5)
plt.savefig("Data/evaluation/common_ancestor_barplot.png", dpi=300)
plt.close()

