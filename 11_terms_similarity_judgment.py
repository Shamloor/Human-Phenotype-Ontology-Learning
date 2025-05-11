import os
import pandas as pd
import llm_utils

# 文件名列表
ANNOTATION_FILES = [
    "rdfs_label.csv",
    "IAO_0000115.csv",
    "hasExactSynonym.csv",
    "hasRelatedSynonym.csv",
    "hasNarrowSynonym.csv"
]

# 文件路径前缀
EVAL_DIR = "Data/evaluation"

# 遍历每个文件
for filename in ANNOTATION_FILES:
    file_path = os.path.join(EVAL_DIR, filename)
    df = pd.read_csv(file_path, sep="|")

    updated_similarity = []

    for index, row in df.iterrows():
        old_value_raw = row["old_value"]
        new_value_raw = row["new_value"]

        # 情况一：两个都为空 → 相似
        if pd.isna(old_value_raw) and pd.isna(new_value_raw):
            similarity = "1"

        # 情况二：只有一个为空 → 不相似
        elif pd.isna(old_value_raw) or pd.isna(new_value_raw):
            similarity = "0"

        # 情况三：两个都非空 → strip 后再判断是否为空字符串
        else:
            old_value = str(old_value_raw).strip()
            new_value = str(new_value_raw).strip()

            if not old_value or not new_value:
                similarity = "0"
            else:
                prompt = (
                    f"Are the following two phrases semantically similar in the medical domain?\n"
                    f"Old: {old_value}\n"
                    f"New: {new_value}\n"
                    f"Answer with 1 for similar, 0 for not similar. Do not output anything other than 0 or 1."
                )

                response = llm_utils.llm_api(prompt).strip()
                if response not in {"0", "1"}:
                    print(f"[Warning] Unexpected response: '{response}' — retrying...")
                    response_retry = llm_utils.llm_api(prompt).strip()
                    if response_retry not in {"0", "1"}:
                        print(f"[Error] Invalid second response: '{response_retry}' — fallback to '0'")
                        similarity = "0"
                    else:
                        similarity = response_retry
                else:
                    similarity = response

        updated_similarity.append(similarity)

    df["similarity"] = updated_similarity
    df.to_csv(file_path, sep="|", index=False)

print("All similarity values have been updated.")
