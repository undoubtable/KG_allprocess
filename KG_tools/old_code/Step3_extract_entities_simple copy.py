import os
import csv
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pipeline_config import STEP2_SENT_TSV, STEP3_ENT_TSV

sent_tsv_path = str(STEP2_SENT_TSV)
output_entity_path = str(STEP3_ENT_TSV)

# ========= 配置区域：改成你自己的路径 =========
# sent_tsv_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step2_output\第一讲_句子列表.tsv"
# output_entity_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step3_output\第一讲_实体列表.tsv"

# NER 模型配置
model_name = "uer/roberta-base-finetuned-cluener2020-chinese"
device = 0              # CPU=-1；如果有 GPU 就写 0
score_threshold = 0.50   # 过滤低置信度实体
min_char_len = 2         # 实体最小长度（基础过滤）

# 中文数字 + 阿拉伯数字，用于过滤“纯数字实体”
CH_NUMERIC_CHARS = set("一二三四五六七八九十百千万零〇0１２３４５６７８９0123456789")


def load_sentences_from_tsv(tsv_path):
    """
    加载 Step2 生成的 TSV：sentence_id | page_no | text
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    sentences = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sentence_id, page_no, text = parts[0], parts[1], parts[2]
            sentences.append(
                {
                    "sentence_id": sentence_id,
                    "page_no": int(page_no),
                    "text": text,
                }
            )
    return sentences


def clean_mention(mention: str) -> str:
    """
    清洗实体字符串：
    - 去掉里面所有空白字符
    - 去掉前后的 《》「」【】[]()（） 等
    - 去掉多余的前导“和”“的”（如果长度足够）
    - 特别处理：刑法修正案(八 -> 刑法修正案(八)
    """
    # 去掉所有空白
    m = re.sub(r"\s+", "", mention)

    # 去掉书名号/括号等外围符号
    m = m.strip("《》「」【】[]()（）\"'“”‘’")

    # 去掉前导“和”“的”（如果实体够长，避免影响诸如“和平方”这种真实体）
    if len(m) > 3 and m[0] in ("和", "的"):
        m = m[1:]

    # ------- 特殊处理：刑法修正案(八 / 刑法修正案（八 -------
    # 1）半角括号
    if m.startswith("刑法修正案(") and not (m.endswith(")") or m.endswith("）")):
        m = m + ")"

    # 2）全角括号
    if m.startswith("刑法修正案（") and not (m.endswith(")") or m.endswith("）")):
        m = m + "）"

    return m



def is_pure_numeric(mention: str) -> bool:
    """
    判断实体是否“纯数字/纯中文数字”：
    例如：十二、一、2010
    """
    if not mention:
        return False
    return all(ch in CH_NUMERIC_CHARS for ch in mention)


def is_bad_mention(mention: str, ent_type: str) -> bool:
    """
    各种“垃圾实体”的过滤规则集合。
    返回 True 表示丢弃。
    """
    # 基础长度过滤
    if len(mention) < min_char_len:
        return True

    # 丢弃纯数字实体（比如 “十二”、“2010”）
    if is_pure_numeric(mention):
        return True

    # 对 book 类实体做更严格一点的过滤
    if ent_type == "book":
        # 很短的 book 实体，一般没什么用
        if len(mention) < 3:
            return True

        # 以“的 / 和”开头且整体很短的，通常是残缺片段，比如“的决定”“和刑法”
        if mention[0] in ("的", "和") and len(mention) <= 4:
            return True

        # 特别排除一些很泛的短词
        if mention in ("决定", "修正案"):
            return True

    return False


def postprocess_entities(raw_entities):
    """
    对模型直接输出的实体做后处理：
    1. sentence 内去掉重叠实体（保留更长 / 置信度更高的）
    2. 全局按 (page_no, mention, ent_type) 去重
    """
    # 1) 句子内部处理：去重叠 + 同句重复
    ents_by_sent = {}
    for e in raw_entities:
        ents_by_sent.setdefault(e["sentence_id"], []).append(e)

    cleaned = []

    for sent_id, ents in ents_by_sent.items():
        # 先按 start_char 排序，长的优先
        ents_sorted = sorted(
            ents,
            key=lambda x: (x["start_char"], -(x["end_char"] - x["start_char"]))
        )

        kept = []
        for e in ents_sorted:
            overlap = False
            for k in kept:
                # 同类型 & span 重叠 → 认为是同一片区域的竞争实体
                if e["ent_type"] == k["ent_type"]:
                    if not (e["end_char"] <= k["start_char"] or e["start_char"] >= k["end_char"]):
                        # 有重叠，比较谁更“好”
                        len_e = e["end_char"] - e["start_char"]
                        len_k = k["end_char"] - k["start_char"]
                        if len_e < len_k:
                            overlap = True
                            break
                        elif len_e == len_k and e["confidence"] <= k["confidence"]:
                            overlap = True
                            break
                        else:
                            # 当前的更好，淘汰旧的
                            kept.remove(k)
                            break
            if not overlap:
                kept.append(e)

        cleaned.extend(kept)

    # 2) 全局按 (page_no, mention, ent_type) 去重
    unique = []
    seen = set()
    for e in cleaned:
        key = (e["page_no"], e["mention"], e["ent_type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    return unique


def run_ner(sentences):
    """
    使用 HuggingFace pipeline 执行中文 NER，并做清洗 & 过滤 & 去重。
    """
    print(f"🔍 正在加载模型：{model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # 新写法，替代 grouped_entities
        device=device,
    )
    print("✅ 模型加载完成！开始抽取实体...\n")

    raw_entities = []
    ent_id = 1

    for s in sentences:
        text = s["text"]
        if not text.strip():
            continue

        results = ner_pipe(text)

        for r in results:
            score = float(r["score"])
            if score < score_threshold:
                continue

            raw_mention = r["word"]
            mention = clean_mention(raw_mention)
            ent_type = r.get("entity_group", "Entity")

            if is_bad_mention(mention, ent_type):
                continue

            raw_entities.append(
                {
                    "entity_id": f"e{ent_id:05d}",
                    "sentence_id": s["sentence_id"],
                    "page_no": s["page_no"],
                    "mention": mention,
                    "start_char": int(r["start"]),
                    "end_char": int(r["end"]),
                    "ent_type": ent_type,
                    "confidence": score,
                }
            )
            ent_id += 1

    print(f"🧹 模型原始实体数：{len(raw_entities)}，开始做重叠/重复过滤...")
    final_entities = postprocess_entities(raw_entities)
    print(f"✅ 过滤后实体数：{len(final_entities)}")

    return final_entities


def save_entities(entities, output_path):
    """
    输出 TSV：
    entity_id | sentence_id | page_no | mention | start_char | end_char | ent_type | confidence
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("entity_id\tsentence_id\tpage_no\tmention\tstart_char\tend_char\tent_type\tconfidence\n")
        for e in entities:
            f.write(
                f"{e['entity_id']}\t{e['sentence_id']}\t{e['page_no']}\t"
                f"{e['mention']}\t{e['start_char']}\t{e['end_char']}\t"
                f"{e['ent_type']}\t{e['confidence']}\n"
            )

    print(f"✅ 实体列表已保存到：{output_path}")
    print(f"📌 共抽取实体数量：{len(entities)}")


def main():
    sentences = load_sentences_from_tsv(sent_tsv_path)
    print(f"📄 已加载句子数量：{len(sentences)}")

    entities = run_ner(sentences)

    # 预览前 10 条
    print("\n📌 实体示例（前10条）：")
    for e in entities[:10]:
        print(
            f"{e['entity_id']}: {e['mention']} "
            f"({e['ent_type']}, p{e['page_no']}, score={e['confidence']:.2f})"
        )

    save_entities(entities, output_entity_path)


if __name__ == "__main__":
    main()
