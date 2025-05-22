import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.nn.functional import cosine_similarity
from embedding_utils import load_model, get_sentence_embedding

# === 模型加载 ===
tokenizer, model, device = load_model()

# === 路径参数 ===
EVAL_DIR = "Data/evaluation"
SIM_PLOT_DIR = "Data/similarity"
os.makedirs(SIM_PLOT_DIR, exist_ok=True)

# === 文件分类 ===
ALWAYS_EMBED = ["rdfs_label.csv", "IAO_0000115.csv"]
OPTIONAL_FILES = [
    "hasExactSynonym.csv",
    "hasRelatedSynonym.csv",
    "hasNarrowSynonym.csv"
]

# === 保存两个分布以用于合并图像 ===
combined_distributions = {}
combined_means = {}

# === 处理固定输出属性（嵌入判断） ===
for filename in ALWAYS_EMBED:
    file_path = os.path.join(EVAL_DIR, filename)
    df = pd.read_csv(file_path, sep="|")

    similarities = []

    for _, row in df.iterrows():
        old_text = row.get("old_value", "")
        new_text = row.get("new_value", "")

        is_old_empty = pd.isna(old_text) or str(old_text).strip() == ""
        is_new_empty = pd.isna(new_text) or str(new_text).strip() == ""

        if is_old_empty or is_new_empty:
            similarities.append(0.0)
            continue

        try:
            emb_old = get_sentence_embedding(str(old_text).strip(), tokenizer, model, device)
            emb_new = get_sentence_embedding(str(new_text).strip(), tokenizer, model, device)
            sim = cosine_similarity(emb_old.unsqueeze(0), emb_new.unsqueeze(0)).item()
            similarities.append(round(sim, 4))
        except Exception as e:
            print(f"[Error] {filename} embedding failed: {e}")
            similarities.append(0.0)

    df["similarity"] = similarities
    df.to_csv(file_path, sep="|", index=False)
    print(f"✅ Similarity written: {filename}")

    combined_distributions[filename] = similarities
    combined_means[filename] = round(np.mean(similarities), 4)

# === 合并绘图：标签 + 定义 两子图 ===
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, filename in enumerate(ALWAYS_EMBED):
    sims = combined_distributions[filename]
    mean_val = combined_means[filename]
    title = "Label (rdfs:label)" if "label" in filename else "Definition (IAO_0000115)"

    ax = axes[i]
    sns.kdeplot(sims, fill=True, color='skyblue', linewidth=1.5, ax=ax)
    ax.axvline(mean_val, color='gray', linestyle='--', linewidth=1.2, label=f"Mean = {mean_val}")
    ax.text(mean_val + 0.01, ax.get_ylim()[1] * 0.8, f"Mean ≈ {mean_val}", fontsize=10, color='black')
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(SIM_PLOT_DIR, "label_definition_similarity_kde.png"), dpi=300)
plt.close()

# === 处理 Synonym 三类属性 ===
all_syn_similarities = []
silence_total = 0
silence_correct = 0

for filename in OPTIONAL_FILES:
    file_path = os.path.join(EVAL_DIR, filename)
    df = pd.read_csv(file_path, sep="|")

    similarities = []

    for _, row in df.iterrows():
        old_text = row.get("old_value", "")
        new_text = row.get("new_value", "")

        is_old_empty = pd.isna(old_text) or str(old_text).strip() == ""
        is_new_empty = pd.isna(new_text) or str(new_text).strip() == ""

        if is_old_empty:
            silence_total += 1
            if is_new_empty:
                silence_correct += 1
            similarities.append(None)
        else:
            if is_new_empty:
                similarities.append(0.0)
            else:
                try:
                    emb_old = get_sentence_embedding(str(old_text).strip(), tokenizer, model, device)
                    emb_new = get_sentence_embedding(str(new_text).strip(), tokenizer, model, device)
                    sim = cosine_similarity(emb_old.unsqueeze(0), emb_new.unsqueeze(0)).item()
                    similarities.append(round(sim, 4))
                except Exception as e:
                    print(f"[Error] {filename} embedding failed: {e}")
                    similarities.append(0.0)

    df["similarity"] = similarities
    df.to_csv(file_path, sep="|", index=False)
    print(f"✅ Similarity written: {filename}")

    all_syn_similarities.extend([s for s in similarities if s is not None])

# === KDE图：Synonym属性相似度分布 ===
if all_syn_similarities:
    mean_syn = round(np.mean(all_syn_similarities), 4)
    plt.figure(figsize=(6.5, 4.5))
    sns.kdeplot(all_syn_similarities, fill=True, color='lightgreen', linewidth=1.5)
    plt.axvline(mean_syn, color='gray', linestyle='--', linewidth=1.2, label=f"Mean = {mean_syn}")
    plt.text(mean_syn + 0.01, plt.ylim()[1] * 0.8, f"Mean ≈ {mean_syn}", fontsize=10, color='black')
    plt.title("Semantic Similarity Distribution: Synonym Properties", fontsize=13)
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SIM_PLOT_DIR, "synonym_similarity_kde.png"), dpi=300)
    plt.close()

# === 静默柱状图 ===
plt.figure(figsize=(4.5, 4.2))
labels = ['Correct Silence', 'Incorrect Output']
values = [silence_correct, silence_total - silence_correct]
colors = ['#46cdcf', '#3d84a8']
bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
plt.title("Silence Accuracy on Synonym Attributes", fontsize=12)
plt.ylabel("Number of Cases", fontsize=11)
plt.ylim(0, max(values) * 1.15)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{int(height)}',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(SIM_PLOT_DIR, "synonym_silence_accuracy_bar.png"), dpi=300)
plt.close()

print("✅ All similarity evaluations and visualizations completed.")



