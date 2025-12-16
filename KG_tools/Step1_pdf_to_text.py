import os
import re
from pdf2image import convert_from_path
import pytesseract
from pipeline_config import PDF_PATH, STEP1_DIR

pdf_path = str(PDF_PATH)
output_dir = str(STEP1_DIR)

# ======== 基本配置 ========
# pdf_path = r"D:\Desktop\KG_allprocess\KG_files\第一讲.pdf"
# output_dir = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step1_output"
os.makedirs(output_dir, exist_ok=True)

# Poppler & Tesseract 的路径照你的来
POPPLER_PATH = r"D:\Download\poppler-25.11.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ 可以控制只处理前 N 页，调试用；设为 None 就是全文件
MAX_PAGES = None   # 调通之后可以改成 None

# ✅ 语言和 OCR 参数（你已经调过）
ocr_lang = "chi_sim"   # 如果你装了 chi_sim 语言包就用这个
ocr_config = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"


def ocr_pdf_to_pages(pdf_path: str, max_pages=None):
    """
    把 PDF 每一页做 OCR，返回一个列表：
    [
        {"page_no": 1, "text": "..."},
        {"page_no": 2, "text": "..."},
        ...
    ]
    """
    print("🌟 开始将 PDF 转成图片并进行 OCR ……")

    # 先把整个 PDF 转成图片列表
    images = convert_from_path(
        pdf_path,
        poppler_path=POPPLER_PATH,
        dpi=600,  # 分辨率高一点，OCR 效果会好一些
    )

    total_pages = len(images)
    print(f"📄 PDF 共 {total_pages} 页")

    if max_pages is not None:
        images = images[:max_pages]
        print(f"⚠ 仅处理前 {max_pages} 页用于测试")

    pages_text = []

    for i, img in enumerate(images):
        page_no = i + 1
        print(f"\n🔍 OCR 识别第 {page_no} 页……")

        # 可以在这里做裁剪（比如只取上半页），现在我们先用整页
        w, h = img.size
        page_region = img  # img.crop((0, 0, w, h))  # 目前就是整页

        # 直接 OCR
        text = pytesseract.image_to_string(
            page_region, lang=ocr_lang, config=ocr_config
        )

        # 简单清洗一下
        text = text.replace("\x0c", " ")   # 去掉 OCR 末尾的换页符
        text = re.sub(r"\s+", " ", text)   # 合并连续空白
        text = text.strip()

        print(f"  ✅ 第 {page_no} 页字符数: {len(text)}")

        pages_text.append(
            {
                "page_no": page_no,
                "text": text,
            }
        )

    return pages_text


def save_pages_to_txt(pages_text, output_txt_path: str):
    """
    把每页文本写入一个 txt 文件，可以保留分页信息。
    """
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for p in pages_text:
            page_no = p["page_no"]
            text = p["text"]
            f.write(f"=== Page {page_no} ===\n")
            f.write(text)
            f.write("\n\n")

    print(f"\n📁 所有 OCR 文本已保存到: {output_txt_path}")


def main():
    # 输出文件名：和 pdf 同名，加一个 _ocr.txt
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_txt_path = os.path.join(output_dir, f"{pdf_name}_ocr.txt")

    # 1）整份 PDF 做 OCR
    pages_text = ocr_pdf_to_pages(pdf_path, max_pages=MAX_PAGES)

    # 2）保存为 txt
    save_pages_to_txt(pages_text, output_txt_path)


if __name__ == "__main__":
    main()
