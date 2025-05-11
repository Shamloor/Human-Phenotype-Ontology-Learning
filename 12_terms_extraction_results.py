import os
import pandas as pd

# 文件名列表
ANNOTATION_FILES = [
    "rdfs_label.csv",
    "IAO_0000115.csv",
    "hasExactSynonym.csv",
    "hasRelatedSynonym.csv",
    "hasNarrowSynonym.csv"
]

EVAL_DIR = "Data/evaluation"

# 初始化总计
total_correct = 0
total_predicted = 0
total_annotated = 0
total_silence_correct = 0
total_silence_total = 0

per_file_stats = []

# 遍历每个文件
for filename in ANNOTATION_FILES:
    file_path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {filename}, skipping...")
        continue

    df = pd.read_csv(file_path, sep="|")

    correct = df[
        (df["old_value"].notna()) & (df["old_value"] != "") &
        (df["new_value"].notna()) & (df["new_value"] != "") &
        (df["similarity"].astype(str).str.strip() == "1")
    ].shape[0]

    predicted = df[(df["new_value"].notna()) & (df["new_value"] != "")].shape[0]
    annotated = df[(df["old_value"].notna()) & (df["old_value"] != "")].shape[0]

    # 静默准确统计
    silence_correct = df[
        (df["old_value"].isna() | (df["old_value"].astype(str).str.strip() == "")) &
        (df["new_value"].isna() | (df["new_value"].astype(str).str.strip() == ""))
        ].shape[0]

    silence_total = df[
        (df["old_value"].isna() | (df["old_value"].astype(str).str.strip() == ""))
    ].shape[0]

    # 累加总数
    total_correct += correct
    total_predicted += predicted
    total_annotated += annotated
    total_silence_correct += silence_correct
    total_silence_total += silence_total

    # 计算单项指标
    precision = correct / predicted if predicted > 0 else 0
    recall = correct / annotated if annotated > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    silence_acc = silence_correct / silence_total if silence_total > 0 else 0

    per_file_stats.append({
        "file": filename,
        "correct": correct,
        "predicted": predicted,
        "annotated": annotated,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "silence_accuracy": round(silence_acc, 4)
    })

# 总体指标
overall_precision = total_correct / total_predicted if total_predicted > 0 else 0
overall_recall = total_correct / total_annotated if total_annotated > 0 else 0
overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0
overall_silence_accuracy = total_silence_correct / total_silence_total if total_silence_total > 0 else 0

# 打印每个文件的评估指标
print("\n=== Evaluation per file ===")
for stat in per_file_stats:
    print(f"\nFile: {stat['file']}")
    print(f"  正确识别术语数: {stat['correct']}")
    print(f"  系统输出术语总数: {stat['predicted']}")
    print(f"  标注术语总数: {stat['annotated']}")
    print(f"  精确率: {stat['precision']:.4f}")
    print(f"  召回率: {stat['recall']:.4f}")
    print(f"  F1值: {stat['f1']:.4f}")
    print(f"  静默准确率: {stat['silence_accuracy']:.4f}")

# 打印总体指标
print("\n=== Overall Evaluation ===")
print(f"总正确识别术语数: {total_correct}")
print(f"总系统输出术语总数: {total_predicted}")
print(f"总标注术语总数: {total_annotated}")
print(f"总精确率: {overall_precision:.4f}")
print(f"总召回率: {overall_recall:.4f}")
print(f"总F1值: {overall_f1:.4f}")
print(f"总静默准确率: {overall_silence_accuracy:.4f}")

