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
from collections import defaultdict, deque
from math import comb


# ======== 配置路径 ========
NODE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
EDGE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"

# NODE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_nodes_updated.tsv"
# EDGE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_edges_updated.tsv"

# =============== 数据加载 ===============

def load_nodes(path):
    nodes = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            nodes[row["node_id"]] = row
    return nodes


def load_edges(path):
    edges = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                row["confidence"] = float(row.get("confidence", 0.0))
            except:
                row["confidence"] = 0.0
            edges.append(row)
    return edges


# =============== 图结构辅助函数 ===============

def build_adj(nodes, edges):
    adj = {nid: set() for nid in nodes}
    for e in edges:
        u, v = e["src_id"], e["dst_id"]
        if u in adj and v in adj and u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def connected_components(adj):
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


def clustering_coefficient(adj):
    vals = []
    for n, neigh in adj.items():
        k = len(neigh)
        if k < 2:
            continue
        neigh_list = list(neigh)
        tri = sum(
            1
            for i in range(k)
            for j in range(i + 1, k)
            if neigh_list[j] in adj[neigh_list[i]]
        )
        vals.append(tri / comb(k, 2))
    return sum(vals) / len(vals) if vals else 0.0


# =============== 四大维度指标 ===============

# ① Accuracy
def score_accuracy(edges):
    if not edges:
        return 0.0
    return sum(e["confidence"] for e in edges) / len(edges)


# ② Completeness
def score_completeness(nodes, edges, adj):
    n = len(nodes)

    # average degree
    avg_deg = sum(len(adj[nid]) for nid in adj) / n if n else 0.0
    avg_deg_score = min(avg_deg / 10, 1.0)  # normalize

    # clustering coefficient
    cc = clustering_coefficient(adj)

    # largest connected component ratio
    comps = connected_components(adj)
    giant_ratio = max((len(c) for c in comps), default=0) / n if n else 0.0

    completeness = (avg_deg_score + cc + giant_ratio) / 3

    return {
        "avg_degree": avg_deg,
        "avg_degree_score": avg_deg_score,
        "clustering_coefficient": cc,
        "giant_component_ratio": giant_ratio,
        "completeness_score": completeness,
    }


# ③ Consistency
def score_consistency(edges):
    total = len(edges)
    if total == 0:
        return {
            "self_loop_ratio": 0,
            "duplicate_ratio": 0,
            "consistency_score": 1,
        }

    triples = [(e["src_id"], e["dst_id"], e["relation_type"]) for e in edges]
    unique = set(triples)

    duplicates = total - len(unique)
    duplicate_ratio = duplicates / total

    self_loop = sum(1 for e in edges if e["src_id"] == e["dst_id"])
    self_loop_ratio = self_loop / total

    consistency = max(1 - (duplicate_ratio + self_loop_ratio), 0)

    return {
        "self_loop_ratio": self_loop_ratio,
        "duplicate_ratio": duplicate_ratio,
        "consistency_score": consistency,
    }


# ④ Conciseness
def score_conciseness(edges):
    total = len(edges)
    triples = [(e["src_id"], e["dst_id"], e["relation_type"]) for e in edges]
    unique = len(set(triples))
    duplicate_ratio = 1 - unique / total if total else 0
    conciseness = 1 - duplicate_ratio
    return conciseness


# =============== 综合评估 ===============

def evaluate(nodes, edges):
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


# =============== 输出 ===============

def main():
    nodes = load_nodes(NODE_TSV)
    edges = load_edges(EDGE_TSV)

    result = evaluate(nodes, edges)

    # 只输出四大维度 + 综合得分
    print("\n===== KG Quality Evaluation (Zaveri Framework) =====")
    print(f"Accuracy           : {result['accuracy']:.4f}")
    print(f"Completeness       : {result['completeness_score']:.4f}")
    print(f"Consistency        : {result['consistency_score']:.4f}")
    print(f"Conciseness        : {result['conciseness_score']:.4f}")
    print(f"Overall Quality    : {result['overall_quality_score']:.4f}")


if __name__ == "__main__":
    main()
