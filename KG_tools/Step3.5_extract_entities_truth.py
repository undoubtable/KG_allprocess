import os
import json
import yaml
import csv
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from openai import OpenAI

# 你可以在 pipeline_config 里加 truth 输出路径；没有的话用手动路径
from pipeline_config import STEP2_SENT_TSV,STEP35_TRUTH_ENT_TSV

# ========= 路径配置 =========
sent_tsv_path = str(STEP2_SENT_TSV)

# 手动指定 truth 输出（建议你放到 Step3_truth_output 目录）
truth_entity_tsv_path = str(STEP35_TRUTH_ENT_TSV)

# ========= LLM 配置 =========
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

CANDIDATE_MODEL = "DeepSeek-V3"
VERIFY_MODEL = "DeepSeek-R1"

# ========= 控制参数 =========
MAX_CAND_PER_SENT = 12   # 候选可以稍多
MAX_TRUTH_PER_SENT = 6   # truth 更严格更少
DEFAULT_CONF = 0.95      # truth 默认更高（也可以后续用 salience 替代）

LIST_SEPS = ("、", "，", ",", "；", ";", "/", "／")
CH_NUMERIC_CHARS = set("一二三四五六七八九十百千万零〇0１２３４５６７８９0123456789")
BAD_GENERIC = {
    "行为","规定","情况","方面","问题","过程","内容","方式","结果","因素","原则","要求",
    "对象","责任","制度","标准","措施","情形","目的","性质","概念","关系","依据","条件",
    "范围","程度","方法","意见","决定","通知","公告","材料","证据","事实","理由","结论",
}
BAD_SUFFIX = ("方面","问题","情况","过程","内容","方式","结果","因素","原则","要求","制度","关系","标准","措施","情形")
MAX_MENTION_LEN = 20
MIN_LEN = 2


def load_sentences(tsv_path: str) -> List[Dict[str, Any]]:
    sents = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, page_no, text = parts[0], int(parts[1]), parts[2]
            sents.append({"sentence_id": sid, "page_no": page_no, "text": text})
    return sents


def clean_mention(m: str) -> str:
    m = re.sub(r"\s+", "", m).strip()
    m = m.strip("《》「」【】[]()（）\"'“”‘’")
    if len(m) > 3 and m[0] in ("和", "的"):
        m = m[1:]
    if m.startswith("刑法修正案(") and not (m.endswith(")") or m.endswith("）")):
        m += ")"
    if m.startswith("刑法修正案（") and not (m.endswith(")") or m.endswith("）")):
        m += "）"
    return m


def is_pure_numeric(m: str) -> bool:
    return bool(m) and all(ch in CH_NUMERIC_CHARS for ch in m)


def is_bad(m: str) -> bool:
    if len(m) < MIN_LEN or len(m) > MAX_MENTION_LEN:
        return True
    if is_pure_numeric(m):
        return True
    if m in BAD_GENERIC:
        return True
    if m.endswith(BAD_SUFFIX) and len(m) <= 6:
        return True
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", m):
        return True
    return False


def split_list(m: str) -> List[str]:
    if not any(sep in m for sep in LIST_SEPS):
        return [m]
    parts = [m]
    for sep in LIST_SEPS:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts
    parts = [clean_mention(p) for p in parts if p]
    parts = [p for p in parts if not is_bad(p)]
    # 去重保序
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out if out else [m]


def find_span(text: str, m: str) -> Optional[Tuple[int, int]]:
    start = text.find(m)
    if start == -1:
        return None
    return start, start + len(m)


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    l = content.find("{")
    r = content.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(content[l:r + 1])
    except Exception:
        return None


def llm_candidate_entities(text: str) -> List[Dict[str, str]]:
    system = (
        "从句子中抽取实体。输出严格 JSON："
        "{\"entities\":[{\"mention\":\"...\",\"ent_type\":\"...\"},...]}\n"
        "mention 必须是原句连续子串；不得改写。\n"
        "ent_type 只能选：Person, Org, Law, Crime, Location, Time, Book, Concept, Other。\n"
        f"最多输出 {MAX_CAND_PER_SENT} 个，按重要性降序。宁可少抽，不要胡编。"
    )
    resp = client.chat.completions.create(
        model=CANDIDATE_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"句子：{text}"}],
        temperature=0.1,
    )
    data = extract_json((resp.choices[0].message.content or "").strip())
    ents = (data or {}).get("entities", [])
    if not isinstance(ents, list):
        return []
    out = []
    for e in ents:
        if isinstance(e, dict) and e.get("mention"):
            out.append({
                "mention": str(e["mention"]).strip(),
                "ent_type": str(e.get("ent_type", "Other")).strip() or "Other"
            })
    return out[:MAX_CAND_PER_SENT]


def llm_verify_entities(text: str, candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    R1 严格筛选：输出必须是 candidates 的子集
    """
    cand_mentions = [c["mention"] for c in candidates]
    system = (
        "你是实体筛选器。给定句子与候选实体列表，请筛选出应保留的实体。\n"
        "输出严格 JSON：{\"entities\":[{\"mention\":\"...\",\"ent_type\":\"...\"},...]}\n"
        "硬规则：\n"
        "1) mention 必须完全来自候选列表（完全一致）。\n"
        "2) mention 必须是原句连续子串。\n"
        "3) 不要泛化名词（行为/规定/情况/方式/结果/因素/原则/制度/关系等）。\n"
        f"4) 最多保留 {MAX_TRUTH_PER_SENT} 个，按重要性降序。\n"
        "宁可少保留，也不要保留不确定的。"
    )
    resp = client.chat.completions.create(
        model=VERIFY_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"句子：{text}\n候选实体：{cand_mentions}"},
        ],
        temperature=0.1,
    )
    data = extract_json((resp.choices[0].message.content or "").strip())
    ents = (data or {}).get("entities", [])
    if not isinstance(ents, list):
        return []

    # 强约束：必须来自候选 + 必须子串
    cand_set = set(cand_mentions)
    out = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        m = str(e.get("mention", "")).strip()
        if m in cand_set and (text.find(m) != -1):
            out.append({"mention": m, "ent_type": str(e.get("ent_type", "Other")).strip() or "Other"})
    # 去重保序 + 截断
    seen = set()
    dedup = []
    for x in out:
        if x["mention"] not in seen:
            seen.add(x["mention"])
            dedup.append(x)
    return dedup[:MAX_TRUTH_PER_SENT]


def save_tsv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("entity_id\tsentence_id\tpage_no\tmention\tstart_char\tend_char\tent_type\tconfidence\n")
        for r in rows:
            f.write(
                f"{r['entity_id']}\t{r['sentence_id']}\t{r['page_no']}\t{r['mention']}\t"
                f"{r['start_char']}\t{r['end_char']}\t{r['ent_type']}\t{r['confidence']:.4f}\n"
            )
    print(f"✅ Truth 实体已保存：{path}（{len(rows)} 行）")


def main():
    sents = load_sentences(sent_tsv_path)
    print(f"📄 sentences: {len(sents)}")

    out = []
    eid = 1
    seen_global = set()  # (page_no, mention, ent_type)

    for i, s in enumerate(sents, 1):
        text = s["text"]
        if not text.strip():
            continue

        cand = llm_candidate_entities(text)
        # 清洗 + 列表拆分 + 规则过滤 + span 过滤
        cand2 = []
        for c in cand:
            m0 = clean_mention(c["mention"])
            for m in split_list(m0):
                if is_bad(m):
                    continue
                if find_span(text, m) is None:
                    continue
                cand2.append({"mention": m, "ent_type": c["ent_type"]})

        # 去重保序
        seen = set()
        cand3 = []
        for c in cand2:
            if c["mention"] not in seen:
                seen.add(c["mention"])
                cand3.append(c)

        keep = llm_verify_entities(text, cand3)

        for k in keep:
            m = clean_mention(k["mention"])
            sp = find_span(text, m)
            if sp is None:
                continue
            key = (s["page_no"], m, k["ent_type"])
            if key in seen_global:
                continue
            seen_global.add(key)

            out.append({
                "entity_id": f"e{eid:05d}",
                "sentence_id": s["sentence_id"],
                "page_no": s["page_no"],
                "mention": m,
                "start_char": sp[0],
                "end_char": sp[1],
                "ent_type": k["ent_type"],
                "confidence": 0.95,
            })
            eid += 1

        if i % 50 == 0:
            print(f"…processed {i}/{len(sents)}  truth_entities={len(out)}")

    save_tsv(out, truth_entity_tsv_path)


if __name__ == "__main__":
    main()
