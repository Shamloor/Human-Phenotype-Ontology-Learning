import random
sample_size = 100
input_path = "Data/classes/leaf classes.txt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

random_lines = random.sample(lines, sample_size)

filtered_lines = [(line + "\n") for line in random_lines if "HP" in line]

with open("Data/classes/random_leaf_classes.txt", "w", encoding="utf-8") as f:
    f.writelines(filtered_lines)
print(f"处理完成！共保留了 {len(filtered_lines)} 行。")
