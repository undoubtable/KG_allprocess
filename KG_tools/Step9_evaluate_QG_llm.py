#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step9 — Q-version1 评估（KG 驱动 + DeepSeek-R1 评估连贯性）

✅ v2 改动要点：
1) KG 路径优先从 pipeline_config.py 读取 KG_NODES_TSV / KG_EDGES_TSV
   - 若未定义，则 fallback 到 STEP4_NODES_TSV / STEP4_EDGES_TSV（保证可运行）
2) Step8 现在输出 kg_fact/context 后：
   - C（实体对齐）与 D（关系正确）可以严格审计（不再 best-effort）
3) B（连贯性）LLM 调用加 try/except：失败不崩溃，默认给 60 分并写 warning

输出：
- Step9_EVAL_TSV：逐题 A/B/C/D + Q_total
- q_version1_eval_summary.json：整套分布得分 + 均分 + warnings
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from openai import OpenAI
import yaml

from pipeline_config import STEP9_EVAL_TSV, STEP9_EVAL_JSON, STEP8_Q_TSV
from pipeline_config import STEP4_NODES_TSV, STEP4_EDGES_TSV

# ✅ 尝试从 pipeline_config 读取 KG_NODES_TSV / KG_EDGES_TSV（你加了更好；没加也不影响运行）
try:
    from pipeline_config import KG_NODES_TSV as _KG_NODES_TSV
    from pipeline_config import KG_EDGES_TSV as _KG_EDGES_TSV
except Exception:
    _KG_NODES_TSV = STEP4_NODES_TSV
    _KG_EDGES_TSV = STEP4_EDGES_TSV

INPUT_Q_TSV = str(STEP8_Q_TSV)
OUTPUT_EVAL_TSV = str(STEP9_EVAL_TSV)

KG_NODES_TSV = str(_KG_NODES_TSV)
KG_EDGES_TSV = str(_KG_EDGES_TSV)

# ========== LLM 配置 ==========
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

# ✅ 评估推荐用 R1 / reasoner
MODEL_NAME = config.get("judge_model", "DeepSeek-R1")

# ========== 权重与目标分布 ==========
W_A = 0.25
W_B = 0.25
W_C = 0.25
W_D = 0.25

TARGET_ENTITY_DIST = {2: 0.35, 3: 0.45, 4: 0.20}
TARGET_REL_DIST = {1: 1.0}
MIN_ENTITY_LEN = 2


# ================== KG 读取与索引 ==================
def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _read_tsv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return header, rows

def _pick_col(header: List[str], candidates: List[str]) -> Optional[str]:
    hset = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in hset:
            return hset[c.lower()]
    return None

def _detect_nodes_cols(header: List[str]) -> Tuple[str, str, Optional[str]]:
    id_col = _pick_col(header, ["node_id", "id", "entity_id", "uid"]) or header[0]
    name_col = _pick_col(header, ["name", "label", "text", "title", "entity_name"]) or (header[1] if len(header) > 1 else header[0])
    alias_col = _pick_col(header, ["aliases", "alias", "synonyms"])
    return id_col, name_col, alias_col

def _detect_edges_cols(header: List[str]) -> Tuple[str, str, str]:
    src_col = _pick_col(header, ["src_id", "source_id", "source", "head_id", "head", "subj_id", "subj"]) or header[0]
    dst_col = _pick_col(header, ["dst_id", "target_id", "target", "tail_id", "tail", "obj_id", "obj"]) or (header[1] if len(header) > 1 else header[0])
    rel_col = _pick_col(header, ["relation_type", "relation", "rel", "predicate", "edge_type"]) or (header[2] if len(header) > 2 else "relation_type")
    return src_col, dst_col, rel_col

class KGIndex:
    def __init__(self):
        self.id2name: Dict[str, str] = {}
        self.name2ids: Dict[str, Set[str]] = defaultdict(set)
        self.edge_set: Set[Tuple[str, str, str]] = set()

def load_kg(nodes_path: str, edges_path: str) -> KGIndex:
    kg = KGIndex()
    n_header, n_rows = _read_tsv(nodes_path)
    e_header, e_rows = _read_tsv(edges_path)

    n_id_col, n_name_col, n_alias_col = _detect_nodes_cols(n_header)
    e_src_col, e_dst_col, e_rel_col = _detect_edges_cols(e_header)

    for r in n_rows:
        nid = (r.get(n_id_col) or "").strip()
        name = _norm_space(r.get(n_name_col) or "")
        if not nid:
            continue
        if not name:
            name = nid
        kg.id2name[nid] = name
        kg.name2ids[name.lower()].add(nid)

        if n_alias_col:
            aliases = (r.get(n_alias_col) or "").strip()
            if aliases:
                seps = ["|", ",", "，", "；", ";"]
                parts = None
                for sep in seps:
                    if sep in aliases:
                        parts = [_norm_space(x) for x in aliases.split(sep)]
                        break
                if parts is None:
                    parts = [_norm_space(aliases)]
                for a in parts:
                    if a:
                        kg.name2ids[a.lower()].add(nid)

    for r in e_rows:
        s = (r.get(e_src_col) or "").strip()
        d = (r.get(e_dst_col) or "").strip()
        rel = _norm_space(r.get(e_rel_col) or "")
        if s and d and rel:
            kg.edge_set.add((s, rel, d))

    return kg

def build_entity_dict_names(kg: KGIndex, min_len: int = 2) -> List[str]:
    names = [n for n in kg.name2ids.keys() if len(n) >= min_len]
    names.sort(key=len, reverse=True)
    return names

def extract_entities_by_dict(text: str, dict_names_lower: List[str]) -> List[str]:
    t = (text or "").lower()
    found = []
    used = []
    for name in dict_names_lower:
        if name not in t:
            continue
        for m in re.finditer(re.escape(name), t):
            s, e = m.start(), m.end()
            overlap = False
            for ss, ee in used:
                if not (e <= ss or s >= ee):
                    overlap = True
                    break
            if not overlap:
                used.append((s, e))
                found.append(name)
    seen = set()
    out = []
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def parse_kg_fact(text: str) -> Optional[Tuple[str, str, str]]:
    """
    支持：
    - "src_id|rel|dst_id" （推荐）
    - "src_name --rel--> dst_name"
    """
    s = (text or "").strip()
    if not s:
        return None
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
    if "-->" in s and "--" in s:
        m = re.split(r"\s*--\s*", s, maxsplit=1)
        if len(m) == 2:
            src = m[0].strip()
            right = m[1]
            mm = re.split(r"\s*-->\s*", right, maxsplit=1)
            if len(mm) == 2:
                rel = mm[0].strip()
                dst = mm[1].strip()
                if src and rel and dst:
                    return src, rel, dst
    return None


# ================== 题目读取/写出 ==================
def load_mcq(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows

def save_eval_rows(rows: List[Dict[str, Any]], path: str):
    if not rows:
        print("没有数据可写出")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n✅ 评估结果已保存到：{path}（共 {len(rows)} 题）")


# ================== 分布得分（题套级） ==================
def suite_distribution_score(values: List[int], target_dist: Dict[int, float]) -> float:
    if not values:
        return 0.0
    cnt = Counter(values)
    n = sum(cnt.values())
    extra_bins = set(cnt.keys()) - set(target_dist.keys())
    extra_mass = sum(cnt[b] for b in extra_bins) / n if n else 0.0

    l1 = 0.0
    for k, qk in target_dist.items():
        pk = cnt.get(k, 0) / n
        l1 += abs(pk - qk)

    score = 1.0 - 0.5 * l1 - 0.5 * extra_mass
    score = max(0.0, min(1.0, score))
    return score * 100.0

def per_question_A_score(ent_cnt: int, rel_cnt: int, ideal_ent: int = 3, ideal_rel: int = 1) -> float:
    if ent_cnt <= 0:
        ent_score = 0.0
    else:
        dist = abs(ent_cnt - ideal_ent)
        ent_score = max(0.0, 100.0 - dist * 20.0)
    rel_score = 100.0 if rel_cnt >= ideal_rel else 0.0
    return 0.5 * ent_score + 0.5 * rel_score


# ================== C/D 严格校验 ==================
def resolve_ids(kg: KGIndex, raw: str) -> Set[str]:
    if not raw:
        return set()
    if raw in kg.id2name:
        return {raw}
    return set(kg.name2ids.get(raw.lower(), set()))

def score_C_entity_alignment(
    kg: KGIndex,
    dict_names_lower: List[str],
    options: Dict[str, str],
    answer: str,
    fact_trip: Optional[Tuple[str, str, str]],
) -> float:
    """
    有 fact_trip 时：正确选项实体必须匹配 dst（严格）
    """
    if not fact_trip:
        return 0.0

    ans = (answer or "").strip().upper()
    if ans not in options:
        m = re.search(r"\b([ABCD])\b", ans)
        ans = m.group(1) if m else ans
    correct_text = options.get(ans, "")

    _, _, dst_raw = fact_trip
    dst_ids = resolve_ids(kg, dst_raw)

    c_ents = extract_entities_by_dict(correct_text, dict_names_lower)
    c_ids = set()
    for e in c_ents:
        c_ids |= kg.name2ids.get(e.lower(), set())

    if not dst_ids:
        return 0.0
    return 100.0 if (dst_ids & c_ids) else 0.0

def score_D_relation_correctness(
    kg: KGIndex,
    dict_names_lower: List[str],
    options: Dict[str, str],
    answer: str,
    fact_trip: Optional[Tuple[str, str, str]],
) -> float:
    """
    有 fact_trip 时：校验 KG 中是否存在 (src, rel, dst)
    dst 更严格：取 fact dst 与正确选项抽到实体的交集
    """
    if not fact_trip:
        return 0.0

    ans = (answer or "").strip().upper()
    if ans not in options:
        m = re.search(r"\b([ABCD])\b", ans)
        ans = m.group(1) if m else ans
    correct_text = options.get(ans, "")

    src_raw, rel, dst_raw = fact_trip
    src_ids = resolve_ids(kg, src_raw)
    dst_ids_fact = resolve_ids(kg, dst_raw)

    c_ents = extract_entities_by_dict(correct_text, dict_names_lower)
    dst_ids_text = set()
    for e in c_ents:
        dst_ids_text |= kg.name2ids.get(e.lower(), set())

    if dst_ids_fact and dst_ids_text:
        dst_ids = dst_ids_fact & dst_ids_text
        if not dst_ids:
            return 0.0
    else:
        dst_ids = dst_ids_fact or dst_ids_text

    if not src_ids or not dst_ids or not rel:
        return 0.0

    for s in src_ids:
        for d in dst_ids:
            if (s, rel, d) in kg.edge_set:
                return 100.0
    return 0.0


# ================== B：R1 评估连贯性（失败不崩溃） ==================
def parse_percent_score(content: str, key: str, default: int = 60) -> int:
    text = (content or "").strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
            if key in data:
                v = int(data[key])
                return max(0, min(100, v))
    except Exception:
        pass
    return default

def llm_score_coherence(question: str, options: Dict[str, str], answer: str, context: str, kg_fact: str, warnings: List[str]) -> int:
    user_prompt = f"""
你是一名严格的试题审题专家。请只评估“语义连贯性 coherence”这一项，给出 0~100 的整数分。

【题干】
{question}

【选项】
A. {options["A"]}
B. {options["B"]}
C. {options["C"]}
D. {options["D"]}

【正确答案】（仅供评估参考）：{answer}

【原句证据】
{context}

【KG事实】
{kg_fact}

请严格输出 JSON：
{{"coherence_score": 0-100, "issues": ["最多3条"]}}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一名细致严格的中文法律考试命题与审题专家。"},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        return parse_percent_score(content, "coherence_score", default=60)
    except Exception as e:
        warnings.append(f"LLM coherence 评估失败，已使用默认分60。错误：{repr(e)}")
        return 60


# ================== 主流程 ==================
def evaluate_mcq_rows(rows: List[Dict[str, str]], kg: KGIndex, dict_names_lower: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []

    warnings: List[str] = []

    # Step8 v2 应该有 kg_fact/context
    if rows:
        if "kg_fact" not in rows[0]:
            warnings.append("题目 TSV 未包含 kg_fact 列：无法做严格 C/D 校验。")
        if "context" not in rows[0]:
            warnings.append("题目 TSV 未包含 context 列：B 评估仍可做，但可解释性下降。")

    entity_counts: List[int] = []
    relation_counts: List[int] = []

    for idx, r in enumerate(rows, start=1):
        qid = r.get("qid", f"q{idx:04d}")
        question = _norm_space(r.get("question", ""))
        options = {
            "A": _norm_space(r.get("option_a", "")),
            "B": _norm_space(r.get("option_b", "")),
            "C": _norm_space(r.get("option_c", "")),
            "D": _norm_space(r.get("option_d", "")),
        }
        answer = _norm_space(r.get("answer", "")).upper()

        kg_fact = _norm_space(r.get("kg_fact", ""))
        context = _norm_space(r.get("context", ""))

        fact_trip = parse_kg_fact(kg_fact) if kg_fact else None

        # A：实体/关系数量
        full_text = " ".join([question, options["A"], options["B"], options["C"], options["D"]])
        ents = extract_entities_by_dict(full_text, dict_names_lower)
        ent_cnt = len(ents)
        rel_cnt = 1 if fact_trip else 0

        entity_counts.append(ent_cnt)
        relation_counts.append(rel_cnt)

        A_item = per_question_A_score(ent_cnt, rel_cnt, ideal_ent=3, ideal_rel=1)

        # B：连贯性（R1）
        print(f"评估题目 {qid}（B: coherence）...")
        B_item = llm_score_coherence(question, options, answer, context, kg_fact, warnings)

        # C/D：严格校验
        C_item = score_C_entity_alignment(kg, dict_names_lower, options, answer, fact_trip)
        D_item = score_D_relation_correctness(kg, dict_names_lower, options, answer, fact_trip)

        total = W_A * A_item + W_B * B_item + W_C * C_item + W_D * D_item

        r_out = dict(r)
        r_out["ent_cnt"] = str(ent_cnt)
        r_out["rel_cnt"] = str(rel_cnt)
        r_out["entities_matched"] = json.dumps(ents, ensure_ascii=False)
        r_out["A_entity_relation_coverage_score"] = f"{A_item:.2f}"
        r_out["B_coherence_score"] = f"{B_item:.2f}"
        r_out["C_entity_alignment_score"] = f"{C_item:.2f}"
        r_out["D_relation_correctness_score"] = f"{D_item:.2f}"
        r_out["Q_total"] = f"{total:.2f}"
        evaluated.append(r_out)

    suite_ent_dist = suite_distribution_score(entity_counts, TARGET_ENTITY_DIST)
    suite_rel_dist = suite_distribution_score(relation_counts, TARGET_REL_DIST)

    def _avg(col: str) -> float:
        xs = []
        for rr in evaluated:
            try:
                xs.append(float(rr.get(col, "0")))
            except Exception:
                pass
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_questions_tsv": INPUT_Q_TSV,
        "output_eval_tsv": OUTPUT_EVAL_TSV,
        "kg_nodes_tsv": KG_NODES_TSV,
        "kg_edges_tsv": KG_EDGES_TSV,
        "num_questions": len(rows),
        "warnings": warnings,
        "target_entity_dist": TARGET_ENTITY_DIST,
        "target_rel_dist": TARGET_REL_DIST,
        "suite_entity_distribution_score": round(suite_ent_dist, 2),
        "suite_relation_distribution_score": round(suite_rel_dist, 2),
        "weights": {"A": W_A, "B": W_B, "C": W_C, "D": W_D},
        "avg_A": round(_avg("A_entity_relation_coverage_score"), 2),
        "avg_B": round(_avg("B_coherence_score"), 2),
        "avg_C": round(_avg("C_entity_alignment_score"), 2),
        "avg_D": round(_avg("D_relation_correctness_score"), 2),
        "avg_Q_total": round(_avg("Q_total"), 2),
        "suggested_suite_total_with_distribution": round(
            0.2 * suite_ent_dist + 0.2 * suite_rel_dist + 0.6 * _avg("Q_total"),
            2
        )
    }
    return evaluated, summary


def main():
    if not os.path.exists(INPUT_Q_TSV):
        raise FileNotFoundError(f"找不到输入题目文件：{INPUT_Q_TSV}")
    if not os.path.exists(KG_NODES_TSV):
        raise FileNotFoundError(f"找不到 KG nodes：{KG_NODES_TSV}")
    if not os.path.exists(KG_EDGES_TSV):
        raise FileNotFoundError(f"找不到 KG edges：{KG_EDGES_TSV}")

    rows = load_mcq(INPUT_Q_TSV)
    print(f"共读取到 {len(rows)} 道题目。")

    kg = load_kg(KG_NODES_TSV, KG_EDGES_TSV)
    dict_names_lower = build_entity_dict_names(kg, min_len=MIN_ENTITY_LEN)

    eval_rows, summary = evaluate_mcq_rows(rows, kg, dict_names_lower)

    save_eval_rows(eval_rows, OUTPUT_EVAL_TSV)

    out_json = str(STEP9_EVAL_JSON)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary 已保存：{out_json}")

    if summary.get("warnings"):
        print("\n⚠️ Warnings:")
        for w in summary["warnings"]:
            print(" -", w)


if __name__ == "__main__":
    main()
