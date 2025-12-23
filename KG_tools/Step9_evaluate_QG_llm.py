#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step9 — Q-version1 评估（KG 驱动 + 更强模型评估连贯性）

对 Step8 输出题目 TSV 做四项指标评估（百分制 0~100）：
A) entity_relation_coverage_score：题干+选项中包含的实体/关系数量（并支持“题套分布”得分到 100%）
B) coherence_score：语义连贯性（交给更强模型打分 0~100）
C) entity_alignment_score：题干与选项实体对应正确率（尽量做 100% 程序校验）
D) relation_correctness_score：正确答案实体关系正确率（尽量做 100% 程序校验）

输入：
- Step8 输出 TSV（必需列：qid, question, option_a/b/c/d, answer）
- KG nodes.tsv / edges.tsv（用于实体字典与关系校验）
- 可选列：context、fact（或 kg_fact）：
  * 若提供 fact/kg_fact 且能解析 triple，则 C/D 可做到严格校验；
  * 否则 C/D 将退化为 best-effort（会在 summary warnings 说明）。

输出：
- Step9 输出 TSV：附加四项分数、总分、匹配实体等
- summary JSON：题套分布得分、均分、警告

⚠️注意：
- 模型调用与文件导入按照你现有 Step9 格式：OpenAI(base_url=..., api_key=..., headers...) + pipeline_config 常量。
"""

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from openai import OpenAI
from pipeline_config import (
    STEP4_EDGES_TSV,
    STEP4_NODES_TSV,
    STEP9_EVAL_TSV,
    STEP8_Q_TSV,
)

# ================== 配置区域（按你项目习惯改这里即可） ==================

# Step8 输出 TSV
INPUT_Q_TSV = str(STEP8_Q_TSV)

# Step9 输出评估 TSV
OUTPUT_EVAL_TSV = str(STEP9_EVAL_TSV)

# KG 文件（这里先写相对路径/绝对路径；也可改成 pipeline_config 常量）
KG_NODES_TSV = str(STEP4_NODES_TSV)
KG_EDGES_TSV = str(STEP4_EDGES_TSV)

# 评估模型客户端（与你 Step8/Step9 同风格）
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="YOUR_API_KEY_HERE",
    default_headers={"X-Failover-Enabled": "true"},
)

# 用于评估（更强模型，按你实际可用模型名改）
CHAT_MODEL = "DeepSeek-R1"  # 或者你在 Gitee 上对应的 deepseek-reasoner / 其他名称

# Q-version1 总分权重（可按你需求调整）
W_A = 0.25
W_B = 0.25
W_C = 0.25
W_D = 0.25

# 题套实体数分布目标（示例：2~4 个实体为主）
TARGET_ENTITY_DIST = {2: 0.35, 3: 0.45, 4: 0.20}
# 题套关系数分布目标（单边题一般=1）
TARGET_REL_DIST = {1: 1.0}

MIN_ENTITY_LEN = 2  # 实体字典匹配最短长度（避免“法”“人”等过短误匹配）


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
        self.name2ids: Dict[str, Set[str]] = defaultdict(set)  # lower(name)->ids
        self.edge_set: Set[Tuple[str, str, str]] = set()       # (src_id, rel, dst_id)

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
                # 支持多分隔符
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
    names.sort(key=len, reverse=True)  # 最长优先
    return names

def extract_entities_by_dict(text: str, dict_names_lower: List[str]) -> List[str]:
    t = (text or "").lower()
    found = []
    used = []  # spans
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
    # unique
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
    - "src_id|rel|dst_id"
    - "src_name --rel--> dst_name"
    """
    s = (text or "").strip()
    if not s:
        return None
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
    # arrow
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
    """
    L1 距离映射到 0~100：
    score = (1 - 0.5 * sum(|p_i - q_i|) - extra_mass*0.5) * 100
    """
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
    """
    单题 A 分：实体数接近 ideal_ent，且关系数>=ideal_rel
    """
    if ent_cnt <= 0:
        ent_score = 0.0
    else:
        dist = abs(ent_cnt - ideal_ent)
        ent_score = max(0.0, 100.0 - dist * 20.0)  # 偏离每 +1 扣 20

    rel_score = 100.0 if rel_cnt >= ideal_rel else 0.0
    return 0.5 * ent_score + 0.5 * rel_score


# ================== C / D 程序校验（尽量 100%） ==================

def resolve_ids(kg: KGIndex, raw: str) -> Set[str]:
    """
    raw 可能是 node_id 也可能是 name/alias
    """
    if not raw:
        return set()
    if raw in kg.id2name:
        return {raw}
    return set(kg.name2ids.get(raw.lower(), set()))

def score_C_entity_alignment(
    kg: KGIndex,
    dict_names_lower: List[str],
    question: str,
    options: Dict[str, str],
    answer: str,
    fact_trip: Optional[Tuple[str, str, str]],
) -> float:
    """
    C：题干与选项实体对应正确率（0~100）
    - 若 fact_trip 可解析：检查正确选项中是否包含 dst（通过 KG 字典匹配到的 id 与 dst id 交集）
    - 若无 fact_trip：best-effort（正确选项至少能匹配到 KG 实体）
    """
    ans = (answer or "").strip().upper()
    if ans not in options:
        m = re.search(r"\b([ABCD])\b", ans)
        ans = m.group(1) if m else ans
    correct_text = options.get(ans, "")

    if not fact_trip:
        # best-effort：正确选项是否能匹配到任何 KG 实体
        c_ents = extract_entities_by_dict(correct_text, dict_names_lower)
        return 100.0 if c_ents else 0.0

    src_raw, rel_raw, dst_raw = fact_trip
    dst_ids = resolve_ids(kg, dst_raw)

    # 从正确选项抽实体并转 id
    c_ents = extract_entities_by_dict(correct_text, dict_names_lower)
    c_ids = set()
    for e in c_ents:
        c_ids |= kg.name2ids.get(e.lower(), set())

    if not dst_ids:
        # 无法解析 dst，退化处理
        return 50.0

    return 100.0 if (dst_ids & c_ids) else 0.0

def score_D_relation_correctness(
    kg: KGIndex,
    dict_names_lower: List[str],
    options: Dict[str, str],
    answer: str,
    fact_trip: Optional[Tuple[str, str, str]],
) -> float:
    """
    D：正确答案中实体之间关系正确率（0~100）
    - 若 fact_trip 可解析：校验 KG 中是否存在对应 edge
    - src/dst 支持 id 或 name/alias
    - dst 从 “正确选项文本匹配到的实体” 与 “fact_trip 的 dst” 取交集（更严格）
    """
    if not fact_trip:
        return 50.0

    ans = (answer or "").strip().upper()
    if ans not in options:
        m = re.search(r"\b([ABCD])\b", ans)
        ans = m.group(1) if m else ans
    correct_text = options.get(ans, "")

    src_raw, rel, dst_raw = fact_trip
    src_ids = resolve_ids(kg, src_raw)
    dst_ids_fact = resolve_ids(kg, dst_raw)

    # dst 从正确选项抽
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
        return 50.0

    for s in src_ids:
        for d in dst_ids:
            if (s, rel, d) in kg.edge_set:
                return 100.0
    return 0.0


# ================== B：更强模型评估连贯性（0~100） ==================

def parse_percent_score(content: str, key: str, default: int = 60) -> int:
    """
    从 LLM 返回里解析 key 对应的 0~100 分数
    期待 JSON：{"coherence_score": 87, ...}
    """
    result = default
    text = (content or "").strip()

    # JSON 优先
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
            if key in data:
                v = int(data[key])
                if 0 <= v <= 100:
                    return v
    except Exception:
        pass

    # fallback：找数字
    lowered = text.lower()
    idx = lowered.find(key.lower())
    if idx != -1:
        tail = lowered[idx: idx + 80]
        m = re.search(r"(\d{1,3})", tail)
        if m:
            v = int(m.group(1))
            v = max(0, min(100, v))
            result = v

    return result

def llm_score_coherence(
    question: str,
    options: Dict[str, str],
    answer: str,
    context: str = "",
    fact: str = "",
) -> int:
    """
    调用更强模型评估语义连贯性（0~100）
    """
    extra = ""
    if context:
        extra += f"\n【原始上下文】\n{context}\n"
    if fact:
        extra += f"\n【知识图谱事实】\n{fact}\n"

    user_prompt = f"""
你是一名严格的试题审题专家。请只评估“语义连贯性 coherence”这一项，给出 0~100 的整数分。
评估标准：
- 题干是否通顺、信息是否自洽；
- 选项是否与题干在同一语义空间；
- 题干到选项的指向是否清晰，是否存在明显断裂/歧义/不完整导致无法作答。

题目如下：

【题干】
{question}

【选项】
A. {options["A"]}
B. {options["B"]}
C. {options["C"]}
D. {options["D"]}

【正确答案】（仅供你评估参考，不要质疑或修改）：{answer}

{extra}

请严格输出 JSON，不要添加任何其它文字：
{{
  "coherence_score": 分数(0-100),
  "issues": ["最多3条扣分点，短句"]
}}
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一名细致严格的中文法律考试命题与审题专家。"},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    return parse_percent_score(content, "coherence_score", default=60)


# ================== 主流程 ==================

def evaluate_mcq_rows(rows: List[Dict[str, str]], kg: KGIndex, dict_names_lower: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []

    has_context = rows and ("context" in rows[0])
    # 兼容 fact / kg_fact 两种列名
    has_fact = rows and ("fact" in rows[0] or "kg_fact" in rows[0])

    warnings: List[str] = []
    if not os.path.exists(KG_NODES_TSV) or not os.path.exists(KG_EDGES_TSV):
        warnings.append("KG 文件不存在：C/D 关系校验无法执行。")

    if rows:
        if not ("fact" in rows[0] or "kg_fact" in rows[0]):
            warnings.append("题目 TSV 未包含 fact/kg_fact 列：C/D 将退化为 best-effort（不保证100%可审计）。")

    entity_counts = []
    relation_counts = []

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

        context = _norm_space(r.get("context", "")) if has_context else ""
        fact_text = ""
        if has_fact:
            fact_text = _norm_space(r.get("fact", "")) or _norm_space(r.get("kg_fact", ""))

        # 解析 fact triple（若可解析）
        fact_trip = parse_kg_fact(fact_text) if fact_text else None

        # A：实体/关系统计
        full_text = " ".join([question, options["A"], options["B"], options["C"], options["D"]])
        ents = extract_entities_by_dict(full_text, dict_names_lower)
        ent_cnt = len(ents)
        rel_cnt = 1 if fact_trip else 0

        entity_counts.append(ent_cnt)
        relation_counts.append(rel_cnt)

        A_item = per_question_A_score(ent_cnt, rel_cnt, ideal_ent=3, ideal_rel=1)

        # B：更强模型评估连贯性
        print(f"评估题目 {qid}（B: coherence）...")
        B_item = llm_score_coherence(
            question=question,
            options=options,
            answer=answer,
            context=context,
            fact=fact_text,
        )

        # C/D：程序校验
        C_item = score_C_entity_alignment(
            kg=kg,
            dict_names_lower=dict_names_lower,
            question=question,
            options=options,
            answer=answer,
            fact_trip=fact_trip,
        )
        D_item = score_D_relation_correctness(
            kg=kg,
            dict_names_lower=dict_names_lower,
            options=options,
            answer=answer,
            fact_trip=fact_trip,
        )

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

    # 题套分布得分（A 的“依据分布打分100%”更偏题套级）
    suite_ent_dist = suite_distribution_score(entity_counts, TARGET_ENTITY_DIST)
    suite_rel_dist = suite_distribution_score(relation_counts, TARGET_REL_DIST)

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
    }

    # 均分
    def _avg(col: str) -> float:
        xs = []
        for rr in evaluated:
            try:
                xs.append(float(rr.get(col, "0")))
            except Exception:
                pass
        return sum(xs) / len(xs) if xs else 0.0

    summary.update({
        "avg_A": round(_avg("A_entity_relation_coverage_score"), 2),
        "avg_B": round(_avg("B_coherence_score"), 2),
        "avg_C": round(_avg("C_entity_alignment_score"), 2),
        "avg_D": round(_avg("D_relation_correctness_score"), 2),
        "avg_Q_total": round(_avg("Q_total"), 2),
    })

    # 给一个“题套 A 分布融合版”的建议总分（可选）
    summary["suggested_suite_total_with_distribution"] = round(
        0.2 * suite_ent_dist + 0.2 * suite_rel_dist + 0.6 * summary["avg_Q_total"],
        2
    )

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

    # 写出 TSV
    save_eval_rows(eval_rows, OUTPUT_EVAL_TSV)

    # 写出 summary JSON（与 TSV 同目录）
    out_json = os.path.join(os.path.dirname(OUTPUT_EVAL_TSV), "q_version1_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary 已保存：{out_json}")

    if summary.get("warnings"):
        print("\n⚠️ Warnings:")
        for w in summary["warnings"]:
            print(" -", w)


if __name__ == "__main__":
    main()
