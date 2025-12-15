import os
import re

# ======== 配置：改成你自己的路径 ========
ocr_txt_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step1_output\第一讲_ocr.txt"
output_sent_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step2_output\第一讲_句子列表.tsv"


def load_pages_from_ocr_txt(txt_path: str):
    """
    读取形如:
    === Page 1 ===
    这一页的一堆文字...
    === Page 2 ===
    ...
    的 OCR 文本，解析成每页一条记录：
    [
        {"page_no": 1, "text": "..."},
        {"page_no": 2, "text": "..."},
        ...
    ]
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    pages = []
    current_page = None
    current_text_lines = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # 匹配 "=== Page X ==="
            m = re.match(r"=== Page\s+(\d+) ===", line)
            if m:
                # 把前一页收尾
                if current_page is not None:
                    full_text = " ".join(current_text_lines)
                    full_text = re.sub(r"\s+", " ", full_text).strip()
                    pages.append(
                        {
                            "page_no": current_page,
                            "text": full_text,
                        }
                    )
                    current_text_lines = []

                current_page = int(m.group(1))
            else:
                # 普通文本行
                if line.strip():
                    current_text_lines.append(line.strip())

    # 收最后一页
    if current_page is not None:
        full_text = " ".join(current_text_lines)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        pages.append(
            {
                "page_no": current_page,
                "text": full_text,
            }
        )

    return pages


def split_to_sentences(pages):
    """
    把每一页的 text 按标点分句，给每句一个 sentence_id。
    返回：
    [
        {"sentence_id": "s0001", "page_no": 1, "text": "..."},
        ...
    ]
    """
    sentences = []
    sent_id = 1

    for page in pages:
        text = page["text"]
        page_no = page["page_no"]

        if not text:
            continue

        tmp = ""
        for ch in text:
            tmp += ch
            # 中英文句号/问号/感叹号都当分句符
            if ch in ("。", "！", "？", ".", "!", "?"):
                s = tmp.strip()
                if s:
                    sentences.append(
                        {
                            "sentence_id": f"s{sent_id:04d}",
                            "page_no": page_no,
                            "text": s,
                        }
                    )
                    sent_id += 1
                tmp = ""

        # 收最后一段（没有标点结尾的）
        if tmp.strip():
            sentences.append(
                {
                    "sentence_id": f"s{sent_id:04d}",
                    "page_no": page_no,
                    "text": tmp.strip(),
                }
            )
            sent_id += 1

    return sentences


def save_sentences_as_tsv(sentences, output_path: str):
    """
    把句子列表保存成一个简单的 TSV（tab 分隔）：
    sentence_id\tpage_no\ttext
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("sentence_id\tpage_no\ttext\n")
        for s in sentences:
            line = f"{s['sentence_id']}\t{s['page_no']}\t{s['text']}\n"
            f.write(line)
    print(f"✅ 句子列表已保存：{output_path}")


def main():
    # 1）从 OCR 结果 txt 解析出每页
    pages = load_pages_from_ocr_txt(ocr_txt_path)
    print(f"📄 解析出 {len(pages)} 页")

    # 2）按句子切分
    sentences = split_to_sentences(pages)
    print(f"✂ 共切出 {len(sentences)} 句，前 10 句预览：\n")
    for s in sentences[:10]:
        print(f"[{s['sentence_id']}] (page {s['page_no']}): {s['text']}")

    # 3）保存成 TSV，给后续实体/关系抽取用
    save_sentences_as_tsv(sentences, output_sent_path)


if __name__ == "__main__":
    main()
