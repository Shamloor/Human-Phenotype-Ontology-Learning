import os
import torch
from tqdm import tqdm
from embedding_utils import load_model, get_sentence_embedding

# === 参数设定 ===
input_dir = 'Data/natural_language_answers'
output_path = 'Data/embedding/new_items_embedding.pt'

# === 加载模型 ===
tokenizer, model, device = load_model()

# === 获取所有txt文件路径 ===
txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
txt_files.sort()  # 可选：确保顺序一致

# === 嵌入处理 ===
index2id_list = []
embedding_list = []

for filename in tqdm(txt_files):
    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    embedding = get_sentence_embedding(content, tokenizer, model, device)

    index2id_list.append(os.path.splitext(filename)[0])
    embedding_list.append(embedding)

embedding_tensor = torch.stack(embedding_list)
index2id = {i: eid for i, eid in enumerate(index2id_list)}
id2index = {eid: i for i, eid in enumerate(index2id_list)}

torch.save({
    'embedding_matrix': embedding_tensor,
    'id2index': id2index,
    'index2id': index2id
}, output_path)

print(f"Embedding saved to {output_path}, shape: {embedding_tensor.shape}")
