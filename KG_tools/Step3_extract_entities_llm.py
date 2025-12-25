import os
import re
import csv
import json
import time
import yaml
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

# 如果你本地没有pipeline_config.py，注释掉下面两行，改用手动配置路径
from pipeline_config import STEP2_SENT_TSV, STEP3_ENT_TSV

# ========= 路径配置 =========
sent_tsv_path = str(STEP2_SENT_TSV)
output_entity_path = str(STEP3_ENT_TSV)

# sent_tsv_path = r"D:\...\Step2_output\句子列表.tsv"
# output_entity_path = r"D:\...\Step3_output\实体列表.tsv"
import yaml
# ========= LLM 配置（沿用你的 Gitee AI 调用方式） =========
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

MODEL_NAME = "DeepSeek-V3"  # 实体抽取推荐用 V3

# ========= 抽取与过滤参数 =========
min_char_len = 2

# 每句最多保留多少实体（控量核心）
MAX_ENT_PER_SENT = 8

# LLM 没有稳定 token-level score：这里用默认值；你也可以后续让模型输出 salience 写入 confidence
DEFAULT_CONF = 0.85

# 中文数字 + 阿拉伯数字，用于过滤“纯数字实体”
CH_NUMERIC_CHARS = set("一二三四五六七八九十百千万零〇0１２３４５６７８９0123456789")

# 泛化/噪声词（可持续扩充）
BAD_GENERIC = {
    "行为","规定","情况","方面","问题","过程","内容","方式","结果","因素","原则","要求",
    "对象","责任","制度","标准","措施","情形","目的","性质","概念","关系","依据","条件",
    "范围","程度","方法","意见","决定","通知","公告","材料","证据","事实","理由","结论",
}
BAD_SUFFIX = ("方面","问题","情况","过程","内容","方式","结果","因素","原则","要求","制度","关系","标准","措施","情形")
MAX_MENTION_LEN = 20

# 列表分隔符：用于把“A、B、C”拆成多个实体
LIST_SEPS = ("、", "，", ",", "；", ";", "/", "／")


def load_sentences_from_tsv(tsv_path: str) -> List[Dict[str, Any]]:
    """
    加载 Step2 生成的 TSV：sentence_id | page_no | text
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(tsv_path)

    sentences = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sentence_id, page_no, text = parts[0], parts[1], parts[2]
            sentences.append({"sentence_id": sentence_id, "page_no": int(page_no), "text": text})
    return sentences


def clean_mention(mention: str) -> str:
    """
    清洗实体字符串：
    - 去掉空白
    - 去掉外围符号
    - 去掉前导“和/的”（长度足够时）
    - 特别处理：刑法修正案括号
    """
    m = re.sub(r"\s+", "", mention)
    m = m.strip("《》「」【】[]()（）\"'“”‘’")

    if len(m) > 3 and m[0] in ("和", "的"):
        m = m[1:]

    if m.startswith("刑法修正案(") and not (m.endswith(")") or m.endswith("）")):
        m = m + ")"
    if m.startswith("刑法修正案（") and not (m.endswith(")") or m.endswith("）")):
        m = m + "）"

    return m


def is_pure_numeric(mention: str) -> bool:
    if not mention:
        return False
    return all(ch in CH_NUMERIC_CHARS for ch in mention)


def is_bad_mention(mention: str, ent_type: str) -> bool:
    """
    垃圾实体过滤：长度/纯数字/泛化词/泛化后缀/过长片段
    """
    if len(mention) < min_char_len:
        return True
    if len(mention) > MAX_MENTION_LEN:
        return True
    if is_pure_numeric(mention):
        return True

    # 纯标点或非常规符号
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", mention):
        return True

    # 泛化词（强过滤）
    if mention in BAD_GENERIC:
        return True
    if mention.endswith(BAD_SUFFIX) and len(mention) <= 6:
        return True

    # book 类特殊规则（兼容大小写）
    if ent_type.lower() == "book":
        if len(mention) < 3:
            return True
        if mention[0] in ("的", "和") and len(mention) <= 4:
            return True
        if mention in ("决定", "修正案"):
            return True

    return False


def _find_span(text: str, mention: str) -> Optional[Tuple[int, int]]:
    start = text.find(mention)
    if start == -1:
        return None
    return start, start + len(mention)


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中尽量提取 JSON：取第一个 { 到最后一个 } 之间
    """
    content = content.strip()
    l = content.find("{")
    r = content.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(content[l:r + 1])
    except Exception:
        return None


def _llm_extract_mentions(text: str) -> List[Dict[str, str]]:
    """
    调用 LLM 抽实体：返回 [{'mention': '...', 'ent_type': '...'}, ...]
    """
    system_prompt = (
        "你是中文信息抽取助手。请从给定句子中抽取实体。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"entities\": [{\"mention\": \"...\", \"ent_type\": \"...\"}, ...]}\n"
        "规则：\n"
        "1) mention 必须是原句中的连续子串，原文复制，不得改写/概括。\n"
        "2) 不要输出纯数字（如“2010”“十二”）。\n"
        "3) ent_type 只能从以下集合选一个：Person, Org, Law, Crime, Location, Time, Book, Concept, Other。\n"
        "4) 不要输出泛化名词（如：行为/规定/情况/方面/问题/过程/内容/方式/结果/因素/原则/要求/制度/关系等）。\n"
        f"5) 每个句子最多输出 {MAX_ENT_PER_SENT} 个实体，按重要性从高到低排序。\n"
        "6) 宁可少抽，不要胡编。\n"
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"句子：{text}"},
        ],
        temperature=0.1,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json(content)
    if not data:
        return []

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    out = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        mention = str(e.get("mention", "")).strip()
        ent_type = str(e.get("ent_type", "Other")).strip()
        if mention:
            out.append({"mention": mention, "ent_type": ent_type})
    return out


def _split_list_mention(mention: str, ent_type: str) -> List[str]:
    """
    把类似 '最高人民法院、最高人民检察院' 拆分成多个实体
    - 只在出现 LIST_SEPS 时拆
    - 拆完做 clean_mention
    """
    if not mention:
        return []
    if not any(sep in mention for sep in LIST_SEPS):
        return [mention]

    # 对少数“确实是固定短语”的情况，避免误拆（你可以继续扩充）
    no_split_whitelist = {"罪刑法定原则"}  # 示例
    if mention in no_split_whitelist:
        return [mention]

    parts = [mention]
    for sep in LIST_SEPS:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts

    parts = [clean_mention(p) for p in parts]
    parts = [p for p in parts if p]  # 去空
    # 过滤太短的碎片
    parts = [p for p in parts if len(p) >= min_char_len]
    # 去重保序
    seen = set()
    dedup = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup if dedup else [mention]


def postprocess_entities(raw_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    后处理：
    1) 句子内去重叠（同 ent_type 比长度/置信度）
    2) 全局按 (page_no, mention, ent_type) 去重
    """
    ents_by_sent = defaultdict(list)
    for e in raw_entities:
        ents_by_sent[e["sentence_id"]].append(e)

    cleaned = []
    for sent_id, ents in ents_by_sent.items():
        ents_sorted = sorted(
            ents,
            key=lambda x: (x["start_char"], -(x["end_char"] - x["start_char"]))
        )

        kept = []
        for e in ents_sorted:
            overlap = False
            for k in list(kept):
                if e["ent_type"] == k["ent_type"]:
                    if not (e["end_char"] <= k["start_char"] or e["start_char"] >= k["end_char"]):
                        len_e = e["end_char"] - e["start_char"]
                        len_k = k["end_char"] - k["start_char"]
                        if len_e < len_k:
                            overlap = True
                            break
                        elif len_e == len_k and e["confidence"] <= k["confidence"]:
                            overlap = True
                            break
                        else:
                            kept.remove(k)
                            break
            if not overlap:
                kept.append(e)

        cleaned.extend(kept)

    unique = []
    seen = set()
    for e in cleaned:
        key = (e["page_no"], e["mention"], e["ent_type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)

    return unique


def run_ner(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    使用 LLM 执行实体抽取，并做清洗/过滤/去重，输出字段保持不变
    """
    print(f"🤖 使用 LLM 模型抽实体：{MODEL_NAME}\n")

    raw_entities = []
    ent_id = 1

    for idx, s in enumerate(sentences, start=1):
        text = s["text"]
        if not text.strip():
            continue

        llm_ents = _llm_extract_mentions(text)

        # ✅ 句子级截断：即使模型输出很多，也只保留前 MAX_ENT_PER_SENT 个
        if len(llm_ents) > MAX_ENT_PER_SENT:
            llm_ents = llm_ents[:MAX_ENT_PER_SENT]

        for r in llm_ents:
            raw_mention = r["mention"]
            ent_type = r.get("ent_type", "Other") or "Other"

            # 先清洗
            cleaned_m = clean_mention(raw_mention)

            # ✅ 列表拆分（拆出多个实体）
            mentions = _split_list_mention(cleaned_m, ent_type)

            for mention in mentions:
                # 过滤
                if is_bad_mention(mention, ent_type):
                    continue

                span = _find_span(text, mention)
                if not span:
                    # 模型没遵守“子串”规则或拆分后定位失败，跳过
                    continue

                start_char, end_char = span
                raw_entities.append(
                    {
                        "entity_id": f"e{ent_id:05d}",
                        "sentence_id": s["sentence_id"],
                        "page_no": s["page_no"],
                        "mention": mention,
                        "start_char": int(start_char),
                        "end_char": int(end_char),
                        "ent_type": ent_type,
                        "confidence": float(DEFAULT_CONF),
                    }
                )
                ent_id += 1

        if idx % 50 == 0:
            print(f"…已处理 {idx}/{len(sentences)} 句，当前实体数：{len(raw_entities)}")

    print(f"\n🧹 LLM 原始实体数：{len(raw_entities)}，开始做重叠/重复过滤...")
    final_entities = postprocess_entities(raw_entities)
    print(f"✅ 过滤后实体数：{len(final_entities)}")

    return final_entities


def save_entities(entities: List[Dict[str, Any]], output_path: str) -> None:
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
                f"{e['ent_type']}\t{e['confidence']:.4f}\n"
            )

    print(f"✅ 实体列表已保存到：{output_path}")
    print(f"📌 共抽取实体数量：{len(entities)}")


def main():
    sentences = load_sentences_from_tsv(sent_tsv_path)
    print(f"📄 已加载句子数量：{len(sentences)}")

    entities = run_ner(sentences)

    print("\n📌 实体示例（前10条）：")
    for e in entities[:10]:
        print(f"{e['entity_id']}: {e['mention']} ({e['ent_type']}, p{e['page_no']}, score={e['confidence']:.2f})")

    save_entities(entities, output_entity_path)


if __name__ == "__main__":
    main()
