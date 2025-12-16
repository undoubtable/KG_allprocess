import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
BASE_DIR = Path(__file__).parent

PDF_DIR = Path(os.getenv("KG_files", r"D:\Desktop\KG_allprocess\KG_files")).resolve()

PIPELINE_ENTRY = BASE_DIR / "run_pipeline.py"

def run_one(pdf_path: Path):
    env = os.environ.copy()
    env["KG_PDF_PATH"] = str(pdf_path)
    env["KG_LECTURE"] = pdf_path.stem  # 自动用文件名当 lecture
    # 可选：env["KG_PROJECT_ROOT"] = r"D:\Desktop\KG_allprocess"

    print(f"\n=== RUN: {pdf_path.name} ===")
    subprocess.run([PY, str(PIPELINE_ENTRY)], cwd=str(BASE_DIR), env=env, check=True)

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found:", PDF_DIR)
        return

    for pdf in pdfs:
        run_one(pdf)

if __name__ == "__main__":
    main()
