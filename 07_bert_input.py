from collections import defaultdict
import pandas as pd

# 输入输出路径
input_path = 'Data/embedding/annotation.txt'
output_path = 'Data/embedding/annotation_natural.csv'

# 收集每个实体的注释信息
entity_annotations = defaultdict(lambda: defaultdict(list))
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(' ', 2)
        if len(parts) == 3:
            entity, prop, value = parts
            entity_annotations[entity][prop].append(value.strip())

# 定义自然语言模板
property_templates = {
    'rdfs:label': lambda vals: f'labeled as "{vals[0]}"' if vals else '',
    'hasExactSynonym': lambda vals: f'also known as {", ".join(["\"" + v + "\"" for v in vals])}' if vals else '',
    'hasRelatedSynonym': lambda vals: f'related terms include {", ".join(["\"" + v + "\"" for v in vals])}' if vals else '',
    'hasNarrowSynonym': lambda vals: f'narrower terms include {", ".join(["\"" + v + "\"" for v in vals])}' if vals else '',
    'IAO_0000115': lambda vals: f'defined as: "{vals[0]}"' if vals else ''
}

# 构建自然语言句子
output_lines = []
for entity, props in entity_annotations.items():
    phrases = []
    for prop, vals in props.items():
        if prop in property_templates:
            phrase = property_templates[prop](vals)
            if phrase:
                phrases.append(phrase)
    if phrases:
        sentence = f"{entity} is " + ', '.join(phrases[:-1])
        if len(phrases) > 1:
            sentence += f', and {phrases[-1]}.'
        else:
            sentence = f"{entity} is {phrases[0]}."
        output_lines.append((entity, sentence))

# 保存到文件
df = pd.DataFrame([{'id': ent, 'sentence': sent} for ent, sent in output_lines])
df.to_csv(output_path, sep='|', index=False, encoding='utf-8', quoting=3)
