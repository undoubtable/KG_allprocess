# Step7_evaluate_kg_final_minimal_cn.py
# -*- coding: utf-8 -*-

"""
最终极简版 Step7（只落盘一张指标表）
- strict：实体用 name+label 完全匹配；关系用三元组完全匹配
- relaxed：实体用“宽松对齐”(exact/substring/similarity)；关系在实体映射后再匹配三元组
- 只输出：STEP7_KG_QUALITY_CSV（可选 STEP7_KG_QUALITY_JSON）
- 指标表包含中文注释 description_cn

Run:
  python Step7_evaluate_kg_final_minimal_cn.py
"""

import os
import json
import pandas as pd
from difflib import SequenceMatcher
from collections import defaultdict

from pipeline_config import (
    STEP12_NODES_TSV,
    STEP12_EDGES_TSV,
    STEP45_NODES_TSV,
    STEP45_EDGES_TSV,
    STEP13_KG_QUALITY_CSV,
    STEP13_KG_QUALITY_JSON,
)

# =========================
# 你可以按需微调的“宽松对齐”参数
# =========================
RELAX_IGNORE_LABEL = True         # True：宽松对齐只看 name_norm，不要求 label 一致（推荐）
RELAX_MATCH_MODE = "hybrid"       # "exact" / "similarity" / "hybrid"
RELAX_SUBSTRING = True            # True：包含关系直接匹配（法律条文类很有效）
RELAX_SIM_THRESHOLD = 0.88        # similarity 阈值（0~1），可根据数据调 0.85~0.92

SAVE_JSON = True                  # 只要 CSV 的话改 False


# =========================
# 基础工具
# =========================
def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def read_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")

def normalize_name(s: str) -> str:
    return " ".join(str(s).strip().split())

def to_float(x: str, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# =========================
# 统一的节点/边解析（兼容你的 Step4/Step4.5 输出）
# =========================
def build_node_table(nodes_df: pd.DataFrame) -> pd.DataFrame:
    df = nodes_df.copy()

    if "name" not in df.columns:
        raise ValueError("Nodes TSV must contain column: name")
    if "label" not in df.columns:
        if "ent_type" in df.columns:
            df["label"] = df["ent_type"]
        else:
            raise ValueError("Nodes TSV must contain column: label (or ent_type)")

    if "node_id" not in df.columns:
        df["node_id"] = [f"n{i:06d}" for i in range(len(df))]

    df["name_norm"] = df["name"].map(normalize_name)
    df["label_norm"] = df["label"].map(lambda x: str(x).strip())
    df["node_key_strict"] = df["name_norm"] + "||" + df["label_norm"]
    df["node_key_relaxed"] = df["name_norm"]  # relaxed 默认只看 name_norm
    return df


def build_edge_table(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    ndf = build_node_table(nodes_df)

    id2name = dict(zip(ndf["node_id"], ndf["name_norm"]))
    id2label = dict(zip(ndf["node_id"], ndf["label_norm"]))
    id2strict = dict(zip(ndf["node_id"], ndf["node_key_strict"]))

    df = edges_df.copy()

    if "src_id" not in df.columns:
        if "source" in df.columns:
            df["src_id"] = df["source"]
        else:
            raise ValueError("Edges TSV must contain src_id (or source)")
    if "dst_id" not in df.columns:
        if "target" in df.columns:
            df["dst_id"] = df["target"]
        else:
            raise ValueError("Edges TSV must contain dst_id (or target)")
    if "relation_type" not in df.columns:
        if "relation" in df.columns:
            df["relation_type"] = df["relation"]
        else:
            raise ValueError("Edges TSV must contain relation_type (or relation)")

    df["src_id"] = df["src_id"].map(lambda x: str(x).strip())
    df["dst_id"] = df["dst_id"].map(lambda x: str(x).strip())
    df["rel"] = df["relation_type"].map(lambda x: str(x).strip())

    # strict endpoints
    df["src_strict"] = df["src_id"].map(lambda x: id2strict.get(x, ""))
    df["dst_strict"] = df["dst_id"].map(lambda x: id2strict.get(x, ""))

    # relaxed endpoints keep name/label (for mapping)
    df["src_name"] = df["src_id"].map(lambda x: id2name.get(x, ""))
    df["dst_name"] = df["dst_id"].map(lambda x: id2name.get(x, ""))
    df["src_label"] = df["src_id"].map(lambda x: id2label.get(x, ""))
    df["dst_label"] = df["dst_id"].map(lambda x: id2label.get(x, ""))

    # drop edges with missing endpoints
    df = df[(df["src_name"] != "") & (df["dst_name"] != "")].copy()

    if "confidence" in df.columns:
        df["conf_f"] = df["confidence"].map(lambda x: to_float(x, 0.0))
    else:
        df["conf_f"] = 0.0

    df["edge_key_strict"] = df["src_strict"] + "||" + df["rel"] + "||" + df["dst_strict"]
    return df


# =========================
# Strict 评估
# =========================
def evaluate_alignment_strict(b_nodes, b_edges, t_nodes, t_edges) -> dict:
    bN = build_node_table(b_nodes)
    tN = build_node_table(t_nodes)
    bE = build_edge_table(b_edges, b_nodes)
    tE = build_edge_table(t_edges, t_nodes)

    set_b_ent = set(bN["node_key_strict"].tolist())
    set_t_ent = set(tN["node_key_strict"].tolist())

    set_b_rel = set(bE["edge_key_strict"].tolist())
    set_t_rel = set(tE["edge_key_strict"].tolist())

    ent = prf(len(set_b_ent & set_t_ent), len(set_b_ent - set_t_ent), len(set_t_ent - set_b_ent))
    rel = prf(len(set_b_rel & set_t_rel), len(set_b_rel - set_t_rel), len(set_t_rel - set_b_rel))

    # 你关心的口径：覆盖度=Recall；正确率=Precision
    derived = {
        "entity_coverage_recall": ent["recall"],
        "relation_coverage_recall": rel["recall"],
        "entity_precision": ent["precision"],
        "relation_precision": rel["precision"],
        "correct_entities_share_in_truth": ent["recall"],
        "correct_edges_share_in_truth": rel["recall"],
        "entity_f1": ent["f1"],
        "relation_f1": rel["f1"],
    }

    return {"entity_overall": ent, "relation_overall": rel, "derived": derived}


# =========================
# Relaxed 评估：宽松实体对齐 + 映射后三元组匹配
# =========================
def build_relaxed_entity_mapping(bN: pd.DataFrame, tN: pd.DataFrame):
    """
    返回 mapping: baseline_entity_key -> truth_entity_key
    - 若 RELAX_IGNORE_LABEL=True，则 entity_key = name_norm
    - 否则 entity_key = name_norm||label_norm
    """
    if RELAX_IGNORE_LABEL:
        b_keys = sorted(set(bN["name_norm"].tolist()), key=lambda x: (-len(x), x))
        t_keys = sorted(set(tN["name_norm"].tolist()), key=lambda x: (-len(x), x))
        b_to_name = {k: k for k in b_keys}
        t_to_name = {k: k for k in t_keys}
    else:
        bN["b_key"] = bN["name_norm"] + "||" + bN["label_norm"]
        tN["t_key"] = tN["name_norm"] + "||" + tN["label_norm"]
        b_keys = sorted(set(bN["b_key"].tolist()), key=lambda x: (-len(x), x))
        t_keys = sorted(set(tN["t_key"].tolist()), key=lambda x: (-len(x), x))
        b_to_name = {k: k.split("||", 1)[0] for k in b_keys}
        t_to_name = {k: k.split("||", 1)[0] for k in t_keys}

    t_set = set(t_keys)

    def try_match(b_key: str):
        b_name = b_to_name[b_key]

        # 1) exact
        if RELAX_MATCH_MODE in ("exact", "hybrid"):
            if RELAX_IGNORE_LABEL:
                if b_name in t_set:
                    return b_name
            else:
                if b_key in t_set:
                    return b_key

        # 2) substring (only meaningful when ignoring label, otherwise too strict)
        if RELAX_MATCH_MODE == "hybrid" and RELAX_SUBSTRING and RELAX_IGNORE_LABEL:
            for t_key in t_keys:
                t_name = t_to_name[t_key]
                if not t_name:
                    continue
                if b_name in t_name or t_name in b_name:
                    return t_key

        # 3) similarity
        if RELAX_MATCH_MODE in ("similarity", "hybrid"):
            best = (None, 0.0)
            for t_key in t_keys:
                t_name = t_to_name[t_key]
                if not t_name:
                    continue
                s = similarity(b_name, t_name)
                if s > best[1]:
                    best = (t_key, s)
            if best[0] is not None and best[1] >= RELAX_SIM_THRESHOLD:
                return best[0]

        return None

    mapping = {}
    used_truth = set()
    for b_key in b_keys:
        m = try_match(b_key)
        if m is None:
            continue
        # 避免过多 many-to-one，先到先得（也可改成允许 many-to-one）
        if m in used_truth:
            continue
        mapping[b_key] = m
        used_truth.add(m)

    # 统计 TP/FP/FN（实体层面）
    tp = set((b, mapping[b]) for b in mapping.keys() if mapping[b] in t_set)
    b_matched = set(b for b, _ in tp)
    t_matched = set(t for _, t in tp)
    fp = set(b_keys) - b_matched
    fn = set(t_keys) - t_matched

    return mapping, tp, fp, fn, set(b_keys), set(t_keys)


def evaluate_alignment_relaxed(b_nodes, b_edges, t_nodes, t_edges) -> dict:
    bN = build_node_table(b_nodes)
    tN = build_node_table(t_nodes)
    bE = build_edge_table(b_edges, b_nodes)
    tE = build_edge_table(t_edges, t_nodes)

    mapping, tp_pairs, fp_ent, fn_ent, b_ent_set, t_ent_set = build_relaxed_entity_mapping(bN, tN)
    ent = prf(len(tp_pairs), len(fp_ent), len(fn_ent))

    # 构建 truth triple set（实体 key 使用和 mapping 同一套）
    if RELAX_IGNORE_LABEL:
        t_triples = set()
        for _, r in tE.iterrows():
            h = r["src_name"]
            tt = r["dst_name"]
            rel = r["rel"]
            if h and tt:
                t_triples.add(h + "||" + rel + "||" + tt)
    else:
        t_triples = set()
        for _, r in tE.iterrows():
            h = r["src_name"] + "||" + r["src_label"]
            tt = r["dst_name"] + "||" + r["dst_label"]
            rel = r["rel"]
            t_triples.add(h + "||" + rel + "||" + tt)

    # baseline triples 映射到 truth key 空间后再比较
    b_triples_mapped = set()
    dropped_unmapped = 0
    for _, r in bE.iterrows():
        rel = r["rel"]
        if RELAX_IGNORE_LABEL:
            bh = r["src_name"]
            bt = r["dst_name"]
            if bh in mapping and bt in mapping:
                h2 = mapping[bh]
                t2 = mapping[bt]
                b_triples_mapped.add(h2 + "||" + rel + "||" + t2)
            else:
                dropped_unmapped += 1
        else:
            bh = r["src_name"] + "||" + r["src_label"]
            bt = r["dst_name"] + "||" + r["dst_label"]
            if bh in mapping and bt in mapping:
                h2 = mapping[bh]
                t2 = mapping[bt]
                b_triples_mapped.add(h2 + "||" + rel + "||" + t2)
            else:
                dropped_unmapped += 1

    rel_tp = len(b_triples_mapped & t_triples)
    rel_fp = len(b_triples_mapped - t_triples)
    rel_fn = len(t_triples - b_triples_mapped)
    rel = prf(rel_tp, rel_fp, rel_fn)

    derived = {
        "entity_coverage_recall": ent["recall"],
        "relation_coverage_recall": rel["recall"],
        "entity_precision": ent["precision"],
        "relation_precision": rel["precision"],
        "correct_entities_share_in_truth": ent["recall"],
        "correct_edges_share_in_truth": rel["recall"],
        "entity_f1": ent["f1"],
        "relation_f1": rel["f1"],
        "relaxed_dropped_edges_unmapped_endpoints": dropped_unmapped,
        "relaxed_mapped_triples_count": len(b_triples_mapped),
        "truth_triples_count": len(t_triples),
    }

    return {"entity_overall": ent, "relation_overall": rel, "derived": derived}


# =========================
# 统一落盘：只保存一张表（含中文注释）
# =========================
def build_metrics_table(strict_res: dict, relaxed_res: dict) -> pd.DataFrame:
    """
    输出列：
    - metric_name: 指标英文 key
    - value: 数值
    - category: strict / relaxed
    - description_cn: 中文注释
    """
    CN = {
        # Entities overall
        "entity_tp": "实体TP数量：baseline与truth匹配到的实体数",
        "entity_fp": "实体FP数量：baseline多抽出的实体数（truth中没有匹配）",
        "entity_fn": "实体FN数量：truth漏掉的实体数（baseline没覆盖）",
        "entity_precision": "实体正确率Precision：baseline抽取实体中有多少比例是对的",
        "entity_recall": "实体覆盖度Recall：truth实体中有多少比例被baseline覆盖",
        "entity_f1": "实体F1：Precision与Recall的调和平均",

        # Relations overall
        "relation_tp": "关系TP数量：baseline与truth匹配到的关系三元组数",
        "relation_fp": "关系FP数量：baseline多抽出的关系三元组（truth中不存在）",
        "relation_fn": "关系FN数量：truth中的关系三元组被baseline漏掉的数量",
        "relation_precision": "关系正确率Precision：baseline抽取关系中有多少比例是对的",
        "relation_recall": "关系覆盖度Recall：truth关系中有多少比例被baseline覆盖",
        "relation_f1": "关系F1：Precision与Recall的调和平均",

        # Derived (user requested)
        "entity_coverage_recall": "实体覆盖度（同 entity_recall）",
        "relation_coverage_recall": "关系覆盖度（同 relation_recall）",
        "correct_entities_share_in_truth": "正确实体在truth中的占比（同实体Recall）",
        "correct_edges_share_in_truth": "正确关系在truth中的占比（同关系Recall）",

        # Relaxed extra
        "relaxed_dropped_edges_unmapped_endpoints": "宽松评估中：由于头尾实体无法对齐而被丢弃的baseline边数量",
        "relaxed_mapped_triples_count": "宽松评估中：baseline三元组在实体映射后可比较的数量",
        "truth_triples_count": "宽松评估中：truth三元组总数量（用于比较）",
    }

    rows = []

    def add_block(prefix: str, res: dict):
        ent = res["entity_overall"]
        rel = res["relation_overall"]
        der = res["derived"]

        # entity overall
        for k in ["tp", "fp", "fn", "precision", "recall", "f1"]:
            metric_name = f"{prefix}_entity_{k}"
            rows.append({
                "metric_name": metric_name,
                "value": float(ent[k]) if isinstance(ent[k], (int, float)) else ent[k],
                "category": prefix,
                "description_cn": CN.get(f"entity_{k}", "")
            })

        # relation overall
        for k in ["tp", "fp", "fn", "precision", "recall", "f1"]:
            metric_name = f"{prefix}_relation_{k}"
            rows.append({
                "metric_name": metric_name,
                "value": float(rel[k]) if isinstance(rel[k], (int, float)) else rel[k],
                "category": prefix,
                "description_cn": CN.get(f"relation_{k}", "")
            })

        # derived: 只保留你关心的那几个 + relaxed 额外诊断（不输出一堆拆分表）
        keep_keys = [
            "entity_coverage_recall",
            "relation_coverage_recall",
            "correct_entities_share_in_truth",
            "correct_edges_share_in_truth",
            # relaxed extra
            "relaxed_dropped_edges_unmapped_endpoints",
            "relaxed_mapped_triples_count",
            "truth_triples_count",
        ]
        for k in keep_keys:
            if k not in der:
                continue
            rows.append({
                "metric_name": f"{prefix}_{k}",
                "value": float(der[k]) if isinstance(der[k], (int, float)) else der[k],
                "category": prefix,
                "description_cn": CN.get(k, "")
            })

    add_block("strict", strict_res)
    add_block("relaxed", relaxed_res)

    return pd.DataFrame(rows)


def main():
    # load
    b_nodes = read_tsv(str(STEP12_NODES_TSV))
    b_edges = read_tsv(str(STEP12_EDGES_TSV))
    t_nodes = read_tsv(str(STEP45_NODES_TSV))
    t_edges = read_tsv(str(STEP45_EDGES_TSV))

    # compute
    strict_res = evaluate_alignment_strict(b_nodes, b_edges, t_nodes, t_edges)
    relaxed_res = evaluate_alignment_relaxed(b_nodes, b_edges, t_nodes, t_edges)

    # build final one-table
    df = build_metrics_table(strict_res, relaxed_res)

    # save
    out_csv = str(STEP13_KG_QUALITY_CSV)
    out_json = str(STEP13_KG_QUALITY_JSON)

    ensure_dir_for_file(out_csv)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if SAVE_JSON:
        ensure_dir_for_file(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    # console summary
    se, sr = strict_res["entity_overall"], strict_res["relation_overall"]
    re, rr = relaxed_res["entity_overall"], relaxed_res["relation_overall"]

    print("\n===== STRICT =====")
    print(f"Entities  P/R/F1: {se['precision']:.4f} / {se['recall']:.4f} / {se['f1']:.4f}")
    print(f"Relations P/R/F1: {sr['precision']:.4f} / {sr['recall']:.4f} / {sr['f1']:.4f}")

    print("\n===== RELAXED =====")
    print(f"Entities  P/R/F1: {re['precision']:.4f} / {re['recall']:.4f} / {re['f1']:.4f}")
    print(f"Relations P/R/F1: {rr['precision']:.4f} / {rr['recall']:.4f} / {rr['f1']:.4f}")

    print("\n[✓] 已保存指标表：")
    print("CSV :", out_csv)
    if SAVE_JSON:
        print("JSON:", out_json)
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
