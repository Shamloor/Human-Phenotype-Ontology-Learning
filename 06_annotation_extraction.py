import os
import subprocess

# 文件路径
annotation_path = 'Data/embedding/annotation.txt'

groovy_path = r"D:\Programming\Groovy\bin\groovy.bat"

for path in [annotation_path]:
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted existing file: {path}")

try:
    print("Running annotation_extraction.groovy ...")
    subprocess.run([groovy_path, "annotation_extraction.groovy"], check=True)

    print("All tasks completed successfully.")

except subprocess.CalledProcessError as e:
    print(f"Groovy script execution failed: {e}")
