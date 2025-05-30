import os
import llm_utils

cq_file = 'Data/CQs/CQs_for_terms_extraction.txt'
input_folder = './Data/topic_identification_answers'
output_folder = './Data/terms_extraction_answers'
os.makedirs(output_folder, exist_ok=True)

with open(cq_file, 'r', encoding='utf-8') as f:
    term_cq_lines = [line.strip() for line in f if line.strip()]

for file_name in sorted(os.listdir(input_folder)):
    if not file_name.endswith('.txt'):
        continue

    file_path = os.path.join(input_folder, file_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        previous_qa_text = f.read().strip()

    prompt = (
        "The following are competency questions and answers from a previous stage.\n"
        "Please analyze them and answer the following questions related to ontology term extraction.\n\n"
        "=== Answers From the Previous Stage ===\n"
        + previous_qa_text + "\n\n"
        "=== Term Extraction Competency Questions ===\n"
    )
    for cq in term_cq_lines:
        prompt += cq + "\n"
    prompt += (
        "\nPlease only output the answers in the following format:\n"
        "1. <Answer to question 1>\n"
        "2. <Answer to question 2>\n"
        "3. hasExactSynonym: <term1>; <term2>\n"
        "   hasRelatedSynonym: <term3>\n"
        "   hasNarrowSynonym: <terms4>\n"
        "(If any synonym type has no appropriate terms, use 'none' as its value.)\n"
        "Do not repeat the questions or add any other commentary.\n"
    )

    response = llm_utils.llm_api(prompt)

    output_path = os.path.join(output_folder, file_name)
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write(response.strip())
    print(f"完成处理：{file_name} -> {output_path}")