import torch

# === 路径设定（根据实际路径修改） ===
embedding_path = 'Data/embedding/new_items_embedding.pt'

# === 加载数据 ===
data = torch.load(embedding_path, map_location='cpu')
index2id = data['index2id']
id2index = data['id2index']

# === 打印前几个映射（可根据需要调整范围） ===
print("📌 前5项 index2id 映射：")
for i in range(min(5, len(index2id))):
    print(f"  index {i} → id {index2id[i]}")

print("\n📌 前5项 id2index 映射：")
for i, (eid, idx) in enumerate(id2index.items()):
    if i >= 5:
        break
    print(f"  id {eid} → index {idx}")

# === 可选：双向一致性验证 ===
print("\n🔁 一致性验证：")
ok = True
for i in range(len(index2id)):
    eid = index2id[i]
    if id2index.get(eid) != i:
        print(f"❌ 不一致: index {i} → id {eid}, 但 id → {id2index.get(eid)}")
        ok = False
        break
if ok:
    print("✅ index2id 与 id2index 映射完全一致")
