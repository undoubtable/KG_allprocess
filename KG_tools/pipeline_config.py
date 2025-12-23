# pipeline_config.py
from pathlib import Path
import os

# 项目根目录：建议就是 KG_allprocess/KG_files 所在的目录
PROJECT_ROOT = Path(os.getenv("KG_PROJECT_ROOT", r"D:\Desktop\KG_allprocess")).resolve()

# 输入 PDF 文件（支持命令行/环境变量覆盖）
PDF_PATH = Path(os.getenv("KG_PDF_PATH", str(PROJECT_ROOT / "PDF_files" / "test1.pdf"))).resolve()

# 讲义名（用于输出文件前缀）
LECTURE = os.getenv("KG_LECTURE", PDF_PATH.stem)  # 默认用pdf文件名当讲义名

# 输出根目录
OUT_ROOT = Path(os.getenv("KG_OUT_ROOT", str(PROJECT_ROOT / "Output"))).resolve()

# 各 Step 输出目录
# Step1
STEP1_DIR = OUT_ROOT / "Step1_output"
STEP1_OCR_TXT = STEP1_DIR / f"{LECTURE}_ocr.txt"

# Step2
STEP2_DIR = OUT_ROOT / "Step2_output"
STEP2_SENT_TSV = STEP2_DIR / f"{LECTURE}_句子列表.tsv"

#Step3
STEP3_DIR = OUT_ROOT / "Step3_output"
STEP3_ENT_TSV  = STEP3_DIR / f"{LECTURE}_实体列表.tsv"

#Step4
STEP4_DIR = OUT_ROOT / "Step4_output"
STEP4_NODES_TSV = STEP4_DIR / f"{LECTURE}_KG_nodes.tsv"
STEP4_EDGES_TSV = STEP4_DIR / f"{LECTURE}_KG_edges.tsv"

# Step5
STEP5_DIR = OUT_ROOT / "Step5_output"
STEP5_GLOBAL_NODES = STEP5_DIR / f"global_{LECTURE}_KG_edges.tsv"
STEP5_GLOBAL_EDGES = STEP5_DIR / f"global_{LECTURE}_KG_edges.tsv"

# Step6
#Step7
STEP7_DIR = OUT_ROOT / "Step7_output"
STEP7_KG_QUALITY_CSV = STEP7_DIR / f"{LECTURE}_KG_quality.csv"
STEP7_KG_QUALITY_JSON = STEP7_DIR / f"{LECTURE}_KG_quality.json"

#Step8
STEP8_DIR = OUT_ROOT / "Step8_output"
PROMPT_PATH = Path(os.getenv("PROMPT_PATH", str(PROJECT_ROOT / "KG_tools" / "prompt.txt"))).resolve()
STEP8_Q_TSV = STEP8_DIR / f"{LECTURE}_MCQ.tsv"

#Step9
STEP9_DIR = OUT_ROOT / "Step9_output"
STEP9_EVAL_TSV = STEP9_DIR / f"{LECTURE}_MCQ_eval.tsv"

#Step10
STEP10_DIR = OUT_ROOT / "Step10_output"
STEP10_Q_REVISED_TSV = STEP10_DIR / f"{LECTURE}_MCQ_auto_revised.tsv"

#Step11
STEP11_DIR = OUT_ROOT / "Step11_output"
STEP11_UPDATE_TSV = STEP11_DIR / f"{LECTURE}_KG_update_suggestions.tsv"

# Step12
STEP12_DIR = OUT_ROOT / "Step12_output"
STEP12_NODES_TSV = STEP12_DIR / f"{LECTURE}_KG_nodes_updated.tsv"
STEP12_EDGES_TSV = STEP12_DIR / f"{LECTURE}_KG_edges_updated.tsv"

#Step13
STEP13_DIR = OUT_ROOT / "Step13_output"
STEP13_KG_QUALITY_CSV = STEP13_DIR / f"{LECTURE}_KG_quality_updated.csv"
STEP13_KG_QUALITY_JSON = STEP13_DIR / f"{LECTURE}_KG_quality_updated.json"

#Step14
STEP14_DIR = OUT_ROOT / "Step14_output"
STEP14_Q_TSV = STEP14_DIR / f"{LECTURE}_MCQ_updated.tsv"

#Step15
STEP15_DIR = OUT_ROOT / "Step15_output"
STEP15_EVAL_TSV = STEP15_DIR / f"{LECTURE}_MCQ_eval_updated.tsv"
