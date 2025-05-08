import torch

# 路径
embedding_path = 'Data/embedding/semantic_embedding.pt'

# 加载文件
data = torch.load(embedding_path, map_location='cpu')

# 解包
embedding_matrix = data['embedding_matrix']
id2index = data['id2index']
index2id = data['index2id']

# 校验结构
assert isinstance(id2index, dict), "id2index 应为 dict[str → int]"
assert isinstance(index2id, dict), "index2id 应为 dict[int → str]"
assert isinstance(embedding_matrix, torch.Tensor), "embedding_matrix 应为 torch.Tensor"

# 维度检查
n_vecs, dim = embedding_matrix.shape
print(f"✅ 向量矩阵维度: {embedding_matrix.shape}")
print(f"✅ id2index 长度: {len(id2index)}")
print(f"✅ index2id 长度: {len(index2id)}")

assert n_vecs == len(id2index) == len(index2id), "数量不一致！"

# 样例验证前5个
print("\n🔍 示例:")
for i in range(5):
    eid = index2id[i]
    idx = id2index[eid]
    vec = embedding_matrix[idx]
    print(f"{i}. {eid} → index: {idx} → vec[:5]: {vec[:5].tolist()}")