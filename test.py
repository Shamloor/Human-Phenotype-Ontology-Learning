import torch

# 加载结构嵌入
structure_data = torch.load('Data/embedding/structure_embedding.pt', map_location='cpu')
structure_vectors = structure_data['model_state_dict']['in_embed.weight']
structure_index2id = structure_data['idx2word']

print(f"结构嵌入 shape: {structure_vectors.shape}")
print(f"结构嵌入 示例 id: {structure_index2id[0]}")
print(f"结构嵌入 示例向量: {structure_vectors[0][:10]}")  # 只展示前10维

# 加载语义嵌入
semantic_data = torch.load('Data/embedding/semantic_embedding.pt', map_location='cpu')
semantic_vectors = semantic_data['embedding_matrix']
semantic_index2id = semantic_data['index2id']

print(f"语义嵌入 shape: {semantic_vectors.shape}")
print(f"语义嵌入 示例 id: {semantic_index2id[0]}")
print(f"语义嵌入 示例向量: {semantic_vectors[0][:10]}")  # 只展示前10维