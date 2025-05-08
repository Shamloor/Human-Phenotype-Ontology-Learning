import os
import pandas as pd

# 1. 输入输出路径设置
input_folder = 'Data/terms_extraction_answers'
output_folder = 'Data/natural_language_answers'
os.makedirs(output_folder, exist_ok=True)

# 2. 获取所有输入文件名（只处理 .txt 文件）
input_filenames = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

# 3. 定义自然语言构建函数
def parse_and_naturalize(lines):
    label = lines[0].strip()
    definition = lines[1].strip()
    synonyms = {
        'hasExactSynonym': [],
        'hasRelatedSynonym': [],
        'hasNarrowSynonym': []
    }

    for line in lines[2:]:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            key = key.strip()
            values = [v.strip() for v in value.split(';') if v.strip()]
            if key in synonyms and not (len(values) == 1 and values[0].lower() == 'none'):
                synonyms[key] = values

    phrases = [f'labeled as "{label}"']
    if definition:
        phrases.append(f'defined as: "{definition}"')

    syn_map = {
        'hasExactSynonym': 'also known as',
        'hasRelatedSynonym': 'related terms include',
        'hasNarrowSynonym': 'narrower terms include'
    }

    for syn_type, phrase_intro in syn_map.items():
        terms = synonyms.get(syn_type, [])
        if terms:
            quoted_terms = ', '.join(f'"{term}"' for term in terms)
            phrases.append(f'{phrase_intro} {quoted_terms}')

    if len(phrases) > 1:
        sentence = 'It is ' + ', '.join(phrases[:-1]) + ', and ' + phrases[-1] + '.'
    else:
        sentence = 'It is ' + phrases[0] + '.'
    return sentence

# 4. 批量处理每个文件
for filename in input_filenames:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with open(input_path, 'r', encoding='utf-8') as fin:
        lines = [line for line in fin if line.strip()]

    sentence = parse_and_naturalize(lines)

    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(sentence)
