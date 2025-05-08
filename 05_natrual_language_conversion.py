import os
import pandas as pd

input_folder = 'Data/terms_extraction_answers'
output_folder = 'Data/natural_language_answers'
os.makedirs(output_folder, exist_ok=True)

input_filenames = [f for f in os.listdir(input_folder) if f.endswith('.txt')]


def clean_prefix(text):
    parts = text.strip().split('.', 1)
    return parts[1].strip() if len(parts) == 2 and parts[0].isdigit() else text.strip()


def parse_and_naturalize(lines):
    label = clean_prefix(lines[0])
    definition = clean_prefix(lines[1])

    synonyms = {
        'hasExactSynonym': [],
        'hasRelatedSynonym': [],
        'hasNarrowSynonym': []
    }

    for line in lines[2:]:
        line = clean_prefix(line)
        if ':' in line:
            key, value = line.split(':', 1)
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


for filename in input_filenames:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with open(input_path, 'r', encoding='utf-8') as fin:
        lines = [line for line in fin if line.strip()]

    sentence = parse_and_naturalize(lines)

    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(sentence)
