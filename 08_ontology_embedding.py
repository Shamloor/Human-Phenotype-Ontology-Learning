import pandas as pd
import torch
from tqdm import tqdm
from embedding_utils import load_model, get_sentence_embedding

# === 参数设定 ===
input_path = 'Data/embedding/annotation_natural.csv'
output_path = 'Data/embedding/ontology_embedding.pt'

# === 加载模型 ===
tokenizer, model, device = load_model()

# === 读取CSV ===
df = pd.read_csv(input_path, sep='|')

# === 嵌入处理 ===
index2id_list = []
embedding_list = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    entity_id = row['id']
    sentence = row['sentence']
    embedding = get_sentence_embedding(sentence, tokenizer, model, device)
    index2id_list.append(entity_id)
    embedding_list.append(embedding)

# === 保存 ===
embedding_tensor = torch.stack(embedding_list)
index2id = {i: eid for i, eid in enumerate(index2id_list)}
id2index = {eid: i for i, eid in enumerate(index2id_list)}

torch.save({
    'embedding_matrix': embedding_tensor,
    'id2index': id2index,
    'index2id': index2id
}, output_path)

print(f"✅ Embedding saved to {output_path}, shape: {embedding_tensor.shape}")
