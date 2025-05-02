import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟余弦相似度数据，集中在0.3到0.9之间
similarities = np.random.normal(loc=0.55, scale=0.1, size=1000)
similarities = np.clip(similarities, 0, 1)

# 设置阈值
low_threshold = 0.43
high_threshold = 0.67

# 绘图
plt.figure(figsize=(10, 6))
plt.hist(similarities, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
plt.axvline(low_threshold, color='red', linestyle='--', label=f'Low Threshold = {low_threshold}')
plt.axvline(high_threshold, color='green', linestyle='--', label=f'High Threshold = {high_threshold}')
plt.title('Distribution of Cosine Similarities with Thresholds')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
