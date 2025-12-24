import subprocess
import sys

PYTHON = sys.executable  # 确保使用当前 conda/env 的 python

def run(cmd):
    print(f"\n🚀 Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step11_KG_update_suggestions.py")
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step12_apply_KG_update.py")
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step13_evaluate_newKG_llm.py")
    run(f"{PYTHON} D:\\Desktop\\KG_allprocess\\KG_tools\\Step7_evaluate_kg_llm.py")