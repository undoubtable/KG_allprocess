import subprocess
import sys

PYTHON = sys.executable  # 确保使用当前 conda/env 的 python

def run(cmd):
    print(f"\n🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step8_generate_questions_llm.py")
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step9_evaluate_QG_llm.py")