"""
Step7 — Knowledge Graph Quality Evaluation
(Implemented according to Zaveri et al., JDIQ 2016)

Reference:
Zaveri, A., et al., "A Survey of Quality of Knowledge Graphs",
Journal of Data and Information Quality, 2016.

This module evaluates a KG using the four core quality dimensions
defined in the paper:

1) Accuracy:
    - correctness of triples
    - here approximated by the mean confidence of extracted relations

2) Completeness:
    - structural completeness proxies recommended by the paper:
        * average degree
        * clustering coefficient
        * largest connected component ratio

3) Consistency:
    - KG should not contain:
        * self-loops
        * duplicate triples
    - consistency score = 1 - (self_loop_ratio + duplicate_ratio)

4) Conciseness:
    - minimality of redundant information
    - conciseness score = 1 - duplicate_ratio
"""

import csv
import os
import json
from collections import deque
from math import comb
from pipeline_config import STEP12_NODES_TSV, STEP12_EDGES_TSV
from pipeline_config import STEP13_KG_QUALITY_CSV, STEP13_KG_QUALITY_JSON

NODE_TSV = str(STEP12_NODES_TSV)
EDGE_TSV = str(STEP12_EDGES_TSV)

OUTPUT_CSV = str(STEP13_KG_QUALITY_CSV)
OUTPUT_JSON = str(STEP13_KG_QUALITY_JSON)


# ======== 配置路径 ========
# NODE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
# EDGE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"

# 输出目录/文件（会自动创建目录）
# OUTPUT_DIR = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step7_output"
# OUTPUT_CSV = os.path.join(OUTPUT_DIR, "KG_quality_evaluation.csv")
# OUTPUT_JSON = os.path.join(OUTPUT_DIR, "KG_quality_evaluation.json")


# =============== 数据加载 ===============

def load_nodes(path: str) -> dict:
    nodes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nodes[row["node_id"]] = row
    return nodes


def load_edges(path: str) -> list:
    edges = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                row["confidence"] = float(row.get("confidence", 0.0))
            except Exception:
                row["confidence"] = 0.0
            edges.append(row)
    return edges


# =============== 图结构辅助函数 ===============

def build_adj(nodes: dict, edges: list) -> dict:
    """无向图邻接表（排除自环），用于结构性指标（度/聚类系数/连通分量）。"""
    adj = {nid: set() for nid in nodes}
    for e in edges:
        u, v = e.get("src_id"), e.get("dst_id")
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def connected_components(adj: dict) -> list:
    """返回所有连通分量（每个分量是节点集合）。"""
    visited, comps = set(), []
    for n in adj:
        if n not in visited:
            comp = set()
            q = deque([n])
            visited.add(n)
            while q:
                u = q.popleft()
                comp.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            comps.append(comp)
    return comps


def clustering_coefficient(adj: dict) -> float:
    """平均局部聚类系数（对k<2的点跳过）。"""
    vals = []
    for n, neigh in adj.items():
        k = len(neigh)
        if k < 2:
            continue
        neigh_list = list(neigh)
        tri = 0
        for i in range(k):
            for j in range(i + 1, k):
                if neigh_list[j] in adj[neigh_list[i]]:
                    tri += 1
        vals.append(tri / comb(k, 2))
    return sum(vals) / len(vals) if vals else 0.0


# =============== 四大维度指标 ===============

# ① Accuracy
def score_accuracy(edges: list) -> float:
    """用关系抽取置信度均值近似准确性。"""
    if not edges:
        return 0.0
    return sum(e.get("confidence", 0.0) for e in edges) / len(edges)


# ② Completeness
def score_completeness(nodes: dict, edges: list, adj: dict) -> dict:
    """
    completeness proxies:
      - average degree (normalized)
      - clustering coefficient
      - largest connected component ratio
    """
    n = len(nodes)
    if n == 0:
        return {
            "avg_degree": 0.0,
            "avg_degree_score": 0.0,
            "clustering_coefficient": 0.0,
            "giant_component_ratio": 0.0,
            "completeness_score": 0.0,
        }

    # average degree
    avg_deg = sum(len(adj[nid]) for nid in adj) / n
    avg_deg_score = min(avg_deg / 10, 1.0)  # normalize (可按你的KG规模调整)

    # clustering coefficient
    cc = clustering_coefficient(adj)

    # largest connected component ratio
    comps = connected_components(adj)
    giant_ratio = max((len(c) for c in comps), default=0) / n

    completeness = (avg_deg_score + cc + giant_ratio) / 3

    return {
        "avg_degree": avg_deg,
        "avg_degree_score": avg_deg_score,
        "clustering_coefficient": cc,
        "giant_component_ratio": giant_ratio,
        "completeness_score": completeness,
    }


# ③ Consistency
def score_consistency(edges: list) -> dict:
    """
    consistency:
      - penalize self-loops
      - penalize duplicate triples (src, dst, relation_type)
      score = max(1 - (self_loop_ratio + duplicate_ratio), 0)
    """
    total = len(edges)
    if total == 0:
        return {
            "self_loop_ratio": 0.0,
            "duplicate_ratio": 0.0,
            "consistency_score": 1.0,
        }

    triples = [(e.get("src_id"), e.get("dst_id"), e.get("relation_type")) for e in edges]
    unique = set(triples)

    duplicates = total - len(unique)
    duplicate_ratio = duplicates / total

    self_loop = sum(1 for e in edges if e.get("src_id") == e.get("dst_id"))
    self_loop_ratio = self_loop / total

    consistency = max(1 - (duplicate_ratio + self_loop_ratio), 0.0)

    return {
        "self_loop_ratio": self_loop_ratio,
        "duplicate_ratio": duplicate_ratio,
        "consistency_score": consistency,
    }


# ④ Conciseness
def score_conciseness(edges: list) -> float:
    """conciseness = 1 - duplicate_ratio"""
    total = len(edges)
    if total == 0:
        return 0.0
    triples = [(e.get("src_id"), e.get("dst_id"), e.get("relation_type")) for e in edges]
    unique = len(set(triples))
    duplicate_ratio = 1 - unique / total
    conciseness = 1 - duplicate_ratio
    return conciseness


# =============== 综合评估 ===============

def evaluate(nodes: dict, edges: list) -> dict:
    adj = build_adj(nodes, edges)

    acc = score_accuracy(edges)
    comp = score_completeness(nodes, edges, adj)
    cons = score_consistency(edges)
    conc = score_conciseness(edges)

    # overall score (可自行调整权重)
    overall = (
        0.35 * acc
        + 0.30 * comp["completeness_score"]
        + 0.25 * cons["consistency_score"]
        + 0.10 * conc
    )

    return {
        "accuracy": acc,
        **comp,
        **cons,
        "conciseness_score": conc,
        "overall_quality_score": overall,
    }


# =============== 保存输出 ===============

def save_results_csv(result: dict, path: str) -> None:
    """保存为CSV：每行一个指标(metric,value)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in result.items():
            writer.writerow([k, v])


def save_results_json(result: dict, path: str) -> None:
    """保存为JSON：完整保留所有指标键值。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# =============== 主函数 ===============

def main():
    # 读取数据
    nodes = load_nodes(NODE_TSV)
    edges = load_edges(EDGE_TSV)

    # 计算指标
    result = evaluate(nodes, edges)

    # 控制台输出（四大维度+综合）
    print("\n===== KG Quality Evaluation (Zaveri Framework) =====")
    print(f"Accuracy           : {result['accuracy']:.4f}")
    print(f"Completeness       : {result['completeness_score']:.4f}")
    print(f"Consistency        : {result['consistency_score']:.4f}")
    print(f"Conciseness        : {result['conciseness_score']:.4f}")
    print(f"Overall Quality    : {result['overall_quality_score']:.4f}")

    # 保存全部指标
    save_results_csv(result, OUTPUT_CSV)
    save_results_json(result, OUTPUT_JSON)

    print("\n[✓] Saved all metrics:")
    print(f" - CSV : {OUTPUT_CSV}")
    print(f" - JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
