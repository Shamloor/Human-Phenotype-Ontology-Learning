import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

input_path = 'Data/embedding/annotation_natural.csv'
output_path = 'Data/embedding/semantic_embedding.pt'

# === 加载模型和分词器（推荐使用 GPU）
tokenizer = AutoTokenizer.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
model = AutoModel.from_pretrained("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 读取CSV
df = pd.read_csv(input_path, sep='|')

# === Mean Pooling 函数
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element: [batch, seq_len, hidden]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

# === 嵌入保存容器
index2id_list = []       # index → ID
embedding_list = []      # 每个句子的嵌入向量

# === 逐句处理
for idx, row in tqdm(df.iterrows(), total=len(df)):
    entity_id = row['id']
    sentence = row['sentence']

    # 编码
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 前向传播 + Mean Pooling
    with torch.no_grad():
        outputs = model(**inputs)
        sentence_embedding = mean_pooling(outputs, inputs['attention_mask'])

    # 存储
    index2id_list.append(entity_id)
    embedding_list.append(sentence_embedding.squeeze().cpu())

# === 构建映射与保存
embedding_tensor = torch.stack(embedding_list)
index2id = {i: eid for i, eid in enumerate(index2id_list)}       # int → str
id2index = {eid: i for i, eid in enumerate(index2id_list)}       # str → int

torch.save({
    'embedding_matrix': embedding_tensor,
    'id2index': id2index,
    'index2id': index2id
}, output_path)

print(f"✅ Embedding saved to {output_path}, shape: {embedding_tensor.shape}")
