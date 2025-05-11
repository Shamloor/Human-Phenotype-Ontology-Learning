import pandas as pd
from rdflib import Graph, URIRef
import os
import re

# === 参数映射 ===
ANNOTATION_MAP = {
    "rdfs_label": "http://www.w3.org/2000/01/rdf-schema#label",
    "IAO_0000115": "http://purl.obolibrary.org/obo/IAO_0000115",
    "hasExactSynonym": "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "hasRelatedSynonym": "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "hasNarrowSynonym": "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"
}

# === 删除旧的CSV文件 ===
output_dir = "Data/evaluation"
for key in ANNOTATION_MAP.keys():
    csv_path = os.path.join(output_dir, f"{key}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

# === 加载OWL本体 ===
g = Graph()
g.parse("Data/owl/hp(original).owl", format="xml")

# === 加载 classes_pdf_map.txt，建立 URL -> PMID 映射 ===
uri_to_pmid = {}
with open("Data/classes/classes_pdf_map.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            uri, pmid = parts
            uri_to_pmid[uri] = pmid.replace("PMID:", "")

# === 加载 terms_extraction_answers 所有PMID对应的回答内容 ===
terms_dir = "Data/terms_extraction_answers"
pmid_to_lines = {}

for filename in os.listdir(terms_dir):
    if filename.endswith(".txt"):
        pmid = filename.replace(".txt", "")
        with open(os.path.join(terms_dir, filename), encoding="utf-8") as f:
            pmid_to_lines[pmid] = f.readlines()

# === 遍历每种注释类型并写入CSV ===
for key, annotation_uri in ANNOTATION_MAP.items():
    annotation_ref = URIRef(annotation_uri)
    rows = []

    for uri, pmid in uri_to_pmid.items():
        subject = URIRef(uri)
        index = uri.split("/")[-1]
        old_values = [str(obj) for _, _, obj in g.triples((subject, annotation_ref, None))]
        merged_old_value = ";".join(old_values)

        new_value = ""
        similarity = ""  # 留空，后续填充

        lines = pmid_to_lines.get(pmid, [])
        lines_joined = "\n".join(lines)

        if key == "rdfs_label":
            for line in lines:
                if line.strip().startswith("1. "):
                    new_value = line.strip()[3:].strip()
                    break

        elif key == "IAO_0000115":
            for line in lines:
                if line.strip().startswith("2. "):
                    new_value = line.strip()[3:].strip()
                    break


        else:  # hasExactSynonym / hasRelatedSynonym / hasNarrowSynonym
            found_section = False
            for i, line in enumerate(lines):
                if not found_section:
                    if line.strip().startswith("3."):
                        found_section = True  # 开始处理，包括当前行
                if found_section and f"{key}:" in line:
                    pattern = rf"{key}:\s*(.*)"
                    match = re.search(pattern, line)
                    if match:
                        value = match.group(1).strip()
                        new_value = "" if value.lower() == "none" else value
                    break

        rows.append((index, merged_old_value, new_value, similarity))

    df = pd.DataFrame(rows, columns=["index", "old_value", "new_value", "similarity"])
    df.to_csv(f"{output_dir}/{key}.csv", sep="|", index=False)

