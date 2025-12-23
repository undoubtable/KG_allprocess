# Step5_plot_KG.py
# -*- coding: utf-8 -*-

"""
Direct-run KG plotting script (no Neo4j).
Run:
  python Step5_plot_KG.py

Outputs (in OUT_DIR):
  baseline.png
  truth.png          (only if TRUTH paths provided)
  diff.png           (only if TRUTH paths provided)
  nodes_tp_fp_fn.csv (only if TRUTH paths provided)
  edges_tp_fp_fn.csv (only if TRUTH paths provided)
"""

import os
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pipeline_config import STEP4_NODES_TSV, STEP4_EDGES_TSV, STEP45_NODES_TSV, STEP45_EDGES_TSV
from pipeline_config import STEP5_OUTPUT_DIR

# =========================================================
# ✅ 直接在这里配置你的 TSV 路径（建议用相对路径）
# =========================================================
# 你的测试文件（你已上传的格式完全匹配 node_id/src_id/dst_id/relation_type）
BASELINE_NODES_TSV = STEP4_NODES_TSV
BASELINE_EDGES_TSV = STEP4_EDGES_TSV

# 如果你有 truth，再填上；否则留空即可（脚本会只画 baseline）
TRUTH_NODES_TSV = STEP45_NODES_TSV
TRUTH_EDGES_TSV = STEP45_EDGES_TSV

OUT_DIR = STEP5_OUTPUT_DIR

# 绘图控制（大图建议限制边，避免糊成一团）
MAX_EDGES_FOR_DRAW = 1500     # None 表示不限制
MIN_EDGE_CONF_FOR_DRAW = 0.0
WITH_LABELS = False           # True 会很拥挤，但更直观

# =========================================================
# 工具函数
# =========================================================
def read_tsv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
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

def build_node_table(nodes_df: pd.DataFrame) -> pd.DataFrame:
    df = nodes_df.copy()

    # 必需列
    if "name" not in df.columns:
        raise ValueError("Nodes TSV must contain column: name")
    if "label" not in df.columns:
        if "ent_type" in df.columns:
            df["label"] = df["ent_type"]
        else:
            raise ValueError("Nodes TSV must contain column: label (or ent_type)")

    # node_id 可选：没有就自动生成
    if "node_id" not in df.columns:
        df["node_id"] = [f"n{i:05d}" for i in range(len(df))]

    df["name_norm"] = df["name"].map(normalize_name)
    df["label_norm"] = df["label"].map(lambda x: str(x).strip())
    df["node_key"] = df["name_norm"] + "||" + df["label_norm"]
    return df

def build_edge_table(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    ndf = build_node_table(nodes_df)
    id2key = dict(zip(ndf["node_id"], ndf["node_key"]))

    df = edges_df.copy()

    # 兼容列名
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

    df["rel"] = df["relation_type"].map(lambda x: str(x).strip())

    # 将 src_id/dst_id 映射到 node_key（name||label）
    df["src_key"] = df["src_id"].map(lambda x: id2key.get(str(x).strip(), ""))
    df["dst_key"] = df["dst_id"].map(lambda x: id2key.get(str(x).strip(), ""))

    # ✅ 丢弃指向不存在 node_id 的边（关键：防止产生 malformed edge_key）
    df = df[(df["src_key"] != "") & (df["dst_key"] != "")].copy()

    df["edge_key"] = df["src_key"] + "||" + df["rel"] + "||" + df["dst_key"]

    if "confidence" in df.columns:
        df["conf_f"] = df["confidence"].map(lambda x: to_float(x, 0.0))
    else:
        df["conf_f"] = 0.0

    return df

def build_nx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                   max_edges: int | None,
                   min_conf: float) -> nx.DiGraph:
    G = nx.DiGraph()

    ndf = build_node_table(nodes_df)
    for _, r in ndf.iterrows():
        node_key = r["node_key"]
        G.add_node(node_key, name=r["name_norm"], label=r["label_norm"], node_id=r["node_id"])

    edf = build_edge_table(edges_df, nodes_df)

    # filter by confidence
    edf = edf[edf["conf_f"] >= float(min_conf)].copy()
    edf = edf.sort_values("conf_f", ascending=False)
    if max_edges is not None and len(edf) > max_edges:
        edf = edf.head(max_edges)

    # 去重：同一 (src_key, rel, dst_key) 只保留 max confidence
    best_conf = {}
    for _, r in edf.iterrows():
        ek = r["edge_key"]
        c = float(r["conf_f"])
        if ek not in best_conf or c > best_conf[ek]:
            best_conf[ek] = c

    for ek, c in best_conf.items():
        parts = str(ek).split("||")
        # 合法 edge_key 至少 5 段：[src_name, src_label, rel, dst_name, dst_label]
        if len(parts) < 5:
            continue
        src = parts[0] + "||" + parts[1]
        rel = parts[2]
        dst = parts[3] + "||" + parts[4]
        if src == dst:
            continue
        G.add_edge(src, dst, relation_type=rel, confidence=c)

    return G

def align_tp_fp_fn(b_nodes, t_nodes, b_edges, t_edges):
    bN = build_node_table(b_nodes)
    tN = build_node_table(t_nodes)

    bE = build_edge_table(b_edges, b_nodes)
    tE = build_edge_table(t_edges, t_nodes)

    b_ent = set(bN["node_key"].tolist())
    t_ent = set(tN["node_key"].tolist())

    b_rel = set(bE["edge_key"].tolist())
    t_rel = set(tE["edge_key"].tolist())

    ent_tp = b_ent & t_ent
    ent_fp = b_ent - t_ent
    ent_fn = t_ent - b_ent

    rel_tp = b_rel & t_rel
    rel_fp = b_rel - t_rel
    rel_fn = t_rel - b_rel

    return ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn

def draw_graph_png(G: nx.DiGraph, out_png: str, title: str, with_labels: bool):
    plt.figure(figsize=(16, 12))
    plt.title(title)

    pos = nx.spring_layout(G, seed=42, k=1 / math.sqrt(max(G.number_of_nodes(), 1)))

    labels = nx.get_node_attributes(G, "label")
    label_set = sorted(set(labels.values()))
    label2idx = {lb: i for i, lb in enumerate(label_set)}
    node_colors = [label2idx.get(labels.get(n, ""), 0) for n in G.nodes()]

    deg = dict(G.degree())
    sizes = [220 + 40 * deg.get(n, 0) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.35)

    if with_labels:
        names = nx.get_node_attributes(G, "name")
        nx.draw_networkx_labels(G, pos, labels={k: names.get(k, "") for k in G.nodes()}, font_size=7)

    plt.axis("off")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

# =========================
# ✅ 修复点：diff 图构建强容错
# =========================
def safe_split_node_key(n: str):
    n = "" if n is None else str(n)
    if "||" in n:
        name, label = n.split("||", 1)
        return name, label
    return n, "UNK_LABEL"

def build_diff_graph(ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn) -> nx.DiGraph:
    G = nx.DiGraph()

    def add_node(n, status):
        if not G.has_node(n):
            name, label = safe_split_node_key(n)
            G.add_node(n, name=name, label=label, status=status)
        else:
            old = G.nodes[n].get("status", "UNK")
            if old != "TP" and status == "TP":
                G.nodes[n]["status"] = "TP"

    for n in ent_tp: add_node(n, "TP")
    for n in ent_fp: add_node(n, "FP")
    for n in ent_fn: add_node(n, "FN")

    def add_edge(e_key, status):
        parts = str(e_key).split("||")
        # 合法 edge_key 至少 5 段：[src_name, src_label, rel, dst_name, dst_label]
        if len(parts) < 5:
            return
        src = parts[0] + "||" + parts[1]
        rel = parts[2]
        dst = parts[3] + "||" + parts[4]
        if not G.has_node(src): add_node(src, "UNK")
        if not G.has_node(dst): add_node(dst, "UNK")
        G.add_edge(src, dst, relation_type=rel, status=status)

    for e in rel_tp: add_edge(e, "TP")
    for e in rel_fp: add_edge(e, "FP")
    for e in rel_fn: add_edge(e, "FN")

    return G

def draw_diff_png(G: nx.DiGraph, out_png: str, title: str, with_labels: bool):
    plt.figure(figsize=(16, 12))
    plt.title(title)

    pos = nx.spring_layout(G, seed=7, k=1 / math.sqrt(max(G.number_of_nodes(), 1)))

    s2c = {"TP": 2, "FP": 0, "FN": 1, "UNK": 3}
    node_status = nx.get_node_attributes(G, "status")
    node_colors = [s2c.get(node_status.get(n, "UNK"), 3) for n in G.nodes()]
    sizes = [260 for _ in G.nodes()]

    edge_status = nx.get_edge_attributes(G, "status")
    edge_colors = [s2c.get(edge_status.get((u, v), "UNK"), 3) for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color=edge_colors, alpha=0.45)

    if with_labels:
        names = nx.get_node_attributes(G, "name")
        nx.draw_networkx_labels(G, pos, labels={k: names.get(k, "") for k in G.nodes()}, font_size=7)

    plt.axis("off")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def save_tp_fp_fn_csv(ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    nodes_rows = []
    for k in sorted(ent_tp):
        name, label = safe_split_node_key(k)
        nodes_rows.append({"node_key": k, "name": name, "label": label, "status": "TP"})
    for k in sorted(ent_fp):
        name, label = safe_split_node_key(k)
        nodes_rows.append({"node_key": k, "name": name, "label": label, "status": "FP"})
    for k in sorted(ent_fn):
        name, label = safe_split_node_key(k)
        nodes_rows.append({"node_key": k, "name": name, "label": label, "status": "FN"})
    pd.DataFrame(nodes_rows).to_csv(os.path.join(out_dir, "nodes_tp_fp_fn.csv"), index=False, encoding="utf-8-sig")

    edges_rows = []
    def add_edges(keys, status):
        for e in sorted(keys):
            parts = str(e).split("||")
            if len(parts) < 5:
                edges_rows.append({"edge_key": e, "status": status})
                continue
            src = parts[0] + "||" + parts[1]
            rel = parts[2]
            dst = parts[3] + "||" + parts[4]
            sname, slabel = safe_split_node_key(src)
            dname, dlabel = safe_split_node_key(dst)
            edges_rows.append({
                "edge_key": e,
                "src_name": sname, "src_label": slabel,
                "relation_type": rel,
                "dst_name": dname, "dst_label": dlabel,
                "status": status
            })
    add_edges(rel_tp, "TP")
    add_edges(rel_fp, "FP")
    add_edges(rel_fn, "FN")
    pd.DataFrame(edges_rows).to_csv(os.path.join(out_dir, "edges_tp_fp_fn.csv"), index=False, encoding="utf-8-sig")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    b_nodes = read_tsv(BASELINE_NODES_TSV)
    b_edges = read_tsv(BASELINE_EDGES_TSV)

    # 1) baseline 图
    Gb = build_nx_graph(b_nodes, b_edges, max_edges=MAX_EDGES_FOR_DRAW, min_conf=MIN_EDGE_CONF_FOR_DRAW)
    draw_graph_png(Gb, os.path.join(OUT_DIR, "baseline.png"), "Baseline KG", with_labels=WITH_LABELS)
    print("[✓] baseline.png saved")

    # 2) 如果 truth 没给，就结束（你当前只提供了 test1 基本就是这种情况）
    if not TRUTH_NODES_TSV or not TRUTH_EDGES_TSV:
        print("[i] Truth paths not provided. Skip truth/diff plotting.")
        print("[✓] Done. Output dir:", OUT_DIR)
        return

    t_nodes = read_tsv(TRUTH_NODES_TSV)
    t_edges = read_tsv(TRUTH_EDGES_TSV)

    # 3) truth 图
    Gt = build_nx_graph(t_nodes, t_edges, max_edges=MAX_EDGES_FOR_DRAW, min_conf=MIN_EDGE_CONF_FOR_DRAW)
    draw_graph_png(Gt, os.path.join(OUT_DIR, "truth.png"), "Truth KG", with_labels=WITH_LABELS)
    print("[✓] truth.png saved")

    # 4) diff 图 + CSV
    ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn = align_tp_fp_fn(b_nodes, t_nodes, b_edges, t_edges)
    Gd = build_diff_graph(ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn)
    save_tp_fp_fn_csv(ent_tp, ent_fp, ent_fn, rel_tp, rel_fp, rel_fn, OUT_DIR)
    draw_diff_png(Gd, os.path.join(OUT_DIR, "diff.png"), "Diff/Error KG (TP/FP/FN)", with_labels=WITH_LABELS)
    print("[✓] diff.png + TP/FP/FN CSV saved")

    print("\n[✓] Done. Outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
