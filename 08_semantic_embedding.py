import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

input_path = 'Data/embedding/annotation_natural.csv'
output_path = 'Data/embedding/semantic_embedding.pt'

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取CSV
df = pd.read_csv(input_path, sep='|')

# 准备保存数据
id2index = []
embedding_list = []

# 逐句处理
for idx, row in tqdm(df.iterrows(), total=len(df)):
    entity_id = row['id']
    sentence = row['sentence']

    # 编码
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state

    # Mean Pooling
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    # 存储
    id2index.append(entity_id)
    embedding_list.append(mean_pooled.squeeze().cpu())

# 构建并保存pt结构
embedding_tensor = torch.stack(embedding_list)
index2id = {i: eid for i, eid in enumerate(id2index)}

torch.save({
    'embedding_matrix': embedding_tensor,
    'id2index': id2index,
    'index2id': index2id
}, output_path)

print(f"Saved to {output_path}")
