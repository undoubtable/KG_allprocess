"""
run_pipeline.py
================

自动顺序执行 KG 构建 + 题目生成 + 评估 + 改写 全流程

Step 顺序：
1. PDF → OCR 文本
2. OCR 文本 → 句子
3. 句子 → 实体抽取
4. 实体 → KG（节点+边）
5. 多讲义 KG 合并（可选）
6. KG 导入 Neo4j（可选）
7. KG 质量评估 K1
8. KG + 原句 → MCQ
9. MCQ 质量评估 Q1
10. MCQ 自动改写
11.对修改进行保存
12.基于保存的修改反馈给KG
13.KG 质量评估 K2
14.KG + 原句 → MCQ
15.MCQ 质量评估 Q2
"""

import subprocess
import sys
import time
from pathlib import Path


# ========== 配置区域 ==========

PYTHON_EXE = sys.executable   # 当前 Python 解释器

BASE_DIR = Path(__file__).parent  # KG_code 目录

STEPS = [
    "Step1_pdf_to_text.py",
    "Step2_ocr_text_to_sentences.py",
    "Step3_extract_entities_simple.py",
    "Step4_extract_relations_simple.py",
    "Step5_build_kg.py",
    # "Step6_load_to_neo4j.py",   # ⚠ 如不想每次都写 Neo4j，可注释
    "Step7_evaluate_kg_new.py",
    "Step8_generate_questions_simple.py",
    "Step9_evaluate_questions.py",
    "Step10_edit_questions.py",
    "Step11_generate_kg_update_suggestions.py",
    "Step12_apply_kg_updates.py",
    "Step13_evaluate_kg_new.py",
    "Step14_generate_questions_simple.py",
    "Step15_evaluate_questions.py"
]


# ========== 核心执行函数 ==========

def run_step(step_name: str):
    step_path = BASE_DIR / step_name
    if not step_path.exists():
        raise FileNotFoundError(f"❌ 找不到脚本：{step_path}")

    print("\n" + "=" * 80)
    print(f"🚀 开始执行 {step_name}")
    print("=" * 80)

    start_time = time.time()

    result = subprocess.run(
        [PYTHON_EXE, str(step_path)],
        cwd=str(BASE_DIR),
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ {step_name} 执行失败（耗时 {elapsed:.1f}s）")
        sys.exit(1)

    print(f"\n✅ {step_name} 执行完成（耗时 {elapsed:.1f}s）")


# ========== main ==========

def main():
    print("\n🎯 开始运行 KG 全流程自动化 Pipeline\n")

    total_start = time.time()

    for step in STEPS:
        run_step(step)

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("🎉 全部步骤执行完成！")
    print(f"⏱ 总耗时：{total_time / 60:.1f} 分钟")
    print("=" * 80)


if __name__ == "__main__":
    main()
