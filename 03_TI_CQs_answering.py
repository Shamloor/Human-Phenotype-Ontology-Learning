import os
import fitz
import llm_utils

pdf_folder = './Data/pdf'
cq_file = 'Data/CQs/CQs_for_topic_identification.txt'
output_folder = './Data/topic_identification_answers'

with open(cq_file, 'r', encoding='utf-8') as f:
    cq_lines = [line.strip() for line in f if line.strip()]

pdf_files = sorted([f for f in os.listdir(pdf_folder)])

for pdf_file in pdf_files:
    pdf_id = pdf_file[:-4]
    print(pdf_id)
    pdf_path = os.path.join(pdf_folder, pdf_file)

    with fitz.open(pdf_path) as doc:
        full_text = "\n".join([page.get_text() for page in doc])

    prompt = "Please read the following medical paper and answer the subsequent competency questions.\n\n"
    prompt += "=== Paper Content ===\n" + full_text.strip() + "\n\n"
    prompt += "=== Competency Questions ===\n"
    for cq in cq_lines:
        prompt += cq + "\n"
    prompt += (
        "If the content of the paper provides rich information for a question, please give a detailed answer."
        "If the information is limited or not clearly available, give a brief or negative answer accordingly."
        "Do not fabricate information that is not supported by the paper."
    )
    prompt += (
        "\nPlease only output the answers in the following format:\n"
        "1. <Answer to question 1>\n"
        "2. <Answer to question 2>\n"
        "...\n"
        "Do not repeat the questions or add any other commentary.\n"
    )

    response = llm_utils.llm_api(prompt)

    output_lines = []
    response_lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    cq_dict = {line.split()[0]: line[len(line.split()[0]):].strip() for line in response_lines if line.split()}

    for cq in cq_lines:
        parts = cq.split(maxsplit=1)
        if len(parts) != 2:
            continue
        index, question = parts
        answer = cq_dict.get(index, "[No answer returned]")
        output_lines.append(f"{index} {question}\n{answer}\n")

    output_path = os.path.join(output_folder, f"{pdf_id}.txt")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

    print(f"完成处理：{pdf_file} -> {output_path}")


