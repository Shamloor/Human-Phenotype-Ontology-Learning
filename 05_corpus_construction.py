import os
import subprocess

# 文件路径
structure_path = './DATA/structure.txt'
annotation_path = './DATA/annotation.txt'

groovy_path = r"D:\Programming\Groovy\bin\groovy.bat"

# 删除旧文件
for path in [structure_path, annotation_path]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted existing file: {path}")

# 执行 Groovy 脚本
try:
    print("Running structure_reasoning.groovy ...")
    subprocess.run([groovy_path, "structure_reasoning.groovy"], check=True)

    print("Running annotation_extraction.groovy ...")
    subprocess.run([groovy_path, "annotation_extraction.groovy"], check=True)

    print("All tasks completed successfully.")

except subprocess.CalledProcessError as e:
    print(f"Groovy script execution failed: {e}")
