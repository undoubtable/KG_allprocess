import pandas as pd
import re
import os
from collections import defaultdict

from pipeline_config import (
    STEP4_NODES_TSV,          # KG-version1（被更新）
    STEP4_EDGES_TSV,
    STEP45_NODES_TSV,         # Truth KG（canonical）
    STEP45_EDGES_TSV,
    STEP10_Q_REVISED_TSV,     # Step10 输出：original_question / revised_question / changed / revision_reason
    STEP11_UPDATE_TSV,        # Step11 输出建议
)

# =========================
# 参数（建议先用这些）
# =========================
# feedback-driven 每题最多新增边
MAX_EDGES_PER_QID = 1
MAX_NODES_PER_QID = 2

# safety net：从 Truth 缺口里挑一小部分做保底
SAFETY_RATIO = 0.30
SAFETY_MAX_EDGES = 200
SAFETY_MAX_NODES = 200

# 最稳的保底策略：只补“两端实体都已在 KG 中”的缺口边（不额外引入很多新节点）
SAFETY_ONLY_EDGES_WITH_EXISTING_NODES = True

# =========================
# 规范化（尽量贴近 relaxed 评估）
# =========================
def norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("（", "(").replace("）", ")")
    return s

# =========================
# 加载节点/边
# =========================
def load_nodes(path):
    df = pd.read_csv(path, sep="\t")
    name2id = {norm(r["name"]): r["node_id"] for _, r in df.iterrows()}
    name2orig = {norm(r["name"]): r["name"] for _, r in df.iterrows()}  # canonical 原名保留
    id2name_norm = {r["node_id"]: norm(r["name"]) for _, r in df.iterrows()}
    return df, name2id, name2orig, id2name_norm

def load_edges(path):
    return pd.read_csv(path, sep="\t")

# Truth 有向边索引： (src_name_norm, dst_name_norm, rel) 集合
def build_truth_directed_edge_set(truth_nodes_df, truth_edges_df):
    truth_id2name_norm = {r["node_id"]: norm(r["name"]) for _, r in truth_nodes_df.iterrows()}
    directed = set()
    for _, e in truth_edges_df.iterrows():
        a = truth_id2name_norm.get(e["src_id"], "")
        b = truth_id2name_norm.get(e["dst_id"], "")
        if not a or not b:
            continue
        rel = str(e["relation_type"])
        directed.add((a, b, rel))
    return directed

# Truth 无向 pair -> relation_type(s)：用于 feedback-driven 的 gate（不关心方向时更宽松）
def build_truth_pair2rels(truth_nodes_df, truth_edges_df):
    truth_id2name_norm = {r["node_id"]: norm(r["name"]) for _, r in truth_nodes_df.iterrows()}
    pair2rels = defaultdict(set)
    for _, e in truth_edges_df.iterrows():
        a = truth_id2name_norm.get(e["src_id"], "")
        b = truth_id2name_norm.get(e["dst_id"], "")
        if not a or not b:
            continue
        pair2rels[frozenset([a, b])].add(str(e["relation_type"]))
    return pair2rels

# KG 有向边集合（按 name_norm 对齐）
def build_kg_directed_edge_set_by_name(kg_edges_df, kg_id2name_norm):
    directed = set()
    for _, e in kg_edges_df.iterrows():
        a = kg_id2name_norm.get(e["src_id"], "")
        b = kg_id2name_norm.get(e["dst_id"], "")
        if not a or not b:
            continue
        directed.add((a, b, str(e["relation_type"])))
    return directed

# KG 无向边集合（按 name_norm 对齐）
def build_kg_undirected_edge_set_by_name(kg_edges_df, kg_id2name_norm):
    undirected = set()
    for _, e in kg_edges_df.iterrows():
        a = kg_id2name_norm.get(e["src_id"], "")
        b = kg_id2name_norm.get(e["dst_id"], "")
        if not a or not b:
            continue
        rel = str(e["relation_type"])
        undirected.add((frozenset([a, b]), rel))
    return undirected

# =========================
# 反馈驱动实体抽取（保留 Step11 架构）
# 说明：这里依旧是“子串命中”，你之前用的是这个范式；
# 如果你想进一步稳，可以替换成 regex matcher 版本。
# =========================
def extract_entities(text: str, vocab_norm_set):
    t = norm(text)
    return {n for n in vocab_norm_set if n in t}

# =========================
# canonical name：只要 Truth 有，就强制用 Truth 的原始名字
# =========================
def canonical_name(name_norm: str, truth_name2orig: dict, fallback_orig: str = "") -> str:
    if name_norm in truth_name2orig:
        return truth_name2orig[name_norm]
    return fallback_orig or name_norm

# =========================
# 主流程
# =========================
def main():
    kg_nodes_df, kg_name2id, kg_name2orig, kg_id2name_norm = load_nodes(str(STEP4_NODES_TSV))
    kg_edges_df = load_edges(str(STEP4_EDGES_TSV))

    truth_nodes_df, truth_name2id, truth_name2orig, truth_id2name_norm = load_nodes(str(STEP45_NODES_TSV))
    truth_edges_df = load_edges(str(STEP45_EDGES_TSV))

    # Truth gate 索引
    truth_pair2rels = build_truth_pair2rels(truth_nodes_df, truth_edges_df)
    truth_directed_edges = build_truth_directed_edge_set(truth_nodes_df, truth_edges_df)

    # KG 当前边集合（用于判断缺口）
    kg_directed_edges = build_kg_directed_edge_set_by_name(kg_edges_df, kg_id2name_norm)
    kg_undirected_edges = build_kg_undirected_edge_set_by_name(kg_edges_df, kg_id2name_norm)

    # 抽实体的词表：KG + Truth（覆盖更广）
    vocab_norm = set(kg_name2id.keys()) | set(truth_name2id.keys())

    # 读 Step10 结果
    q_df = pd.read_csv(str(STEP10_Q_REVISED_TSV), sep="\t")
    required_cols = {"qid", "original_question", "revised_question", "changed", "revision_reason"}
    missing = required_cols - set(q_df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    updates = []
    uid = 1

    # =========================================================
    # Part A：feedback-driven（保留主线）
    # =========================================================
    for _, row in q_df.iterrows():
        if str(row["changed"]).lower() != "yes":
            continue

        qid = row["qid"]
        orig_q = str(row["original_question"])
        rev_q = str(row["revised_question"])
        feedback = str(row["revision_reason"])

        orig_ents = extract_entities(orig_q, vocab_norm)
        rev_ents = extract_entities(rev_q, vocab_norm)

        added = list(rev_ents - orig_ents)
        existing = list(rev_ents & orig_ents)

        # --- 先挑边：只在 Truth 支持时才加（无向 gate，宽松些） ---
        chosen_edges = []  # [(a_norm, b_norm, rel)]
        for a in added:
            if len(chosen_edges) >= MAX_EDGES_PER_QID:
                break
            for b in existing:
                key = frozenset([a, b])
                if key in truth_pair2rels:
                    rel = sorted(list(truth_pair2rels[key]))[0]
                    chosen_edges.append((a, b, rel))
                    break

        # --- 只为“参与边”的新增实体补节点（避免孤立节点） ---
        needed_nodes = set()
        for a, b, _ in chosen_edges:
            needed_nodes.add(a)
            needed_nodes.add(b)

        node_cnt = 0
        for n in added:
            if node_cnt >= MAX_NODES_PER_QID:
                break
            if n not in needed_nodes:
                continue
            if n in kg_name2id:
                continue
            if n not in truth_name2id:
                continue  # 严格：新增节点必须 Truth 支持，避免噪声节点

            updates.append({
                "update_id": f"u{uid:05d}",
                "qid": qid,
                "action": "add_node",
                "entity1_name": canonical_name(n, truth_name2orig),
                "entity2_name": "",
                "entity1_id": "",
                "entity2_id": "",
                "relation_type_old": "",
                "relation_type_new": "",
                "reason": "feedback-driven: 新增实体参与新增边，且 Truth 支持、KG 缺失",
                "revision_reason": feedback,
            })
            uid += 1
            node_cnt += 1

        # --- 输出边：实体名强制 canonical，关系类型用 Truth ---
        for a, b, rel in chosen_edges:
            a_can = canonical_name(a, truth_name2orig, kg_name2orig.get(a, a))
            b_can = canonical_name(b, truth_name2orig, kg_name2orig.get(b, b))

            updates.append({
                "update_id": f"u{uid:05d}",
                "qid": qid,
                "action": "add_edge",
                "entity1_name": a_can,
                "entity2_name": b_can,
                "entity1_id": "",
                "entity2_id": "",
                "relation_type_old": "",
                "relation_type_new": rel,
                "reason": "feedback-driven: 实体对在 Truth 中存在关系（type=Truth）",
                "revision_reason": feedback,
            })
            uid += 1

    # 去重索引：防止 safety-net 重复添加
    planned_node_norm = set()
    planned_edge_directed = set()
    planned_edge_undirected = set()

    for u in updates:
        if u["action"] == "add_node":
            planned_node_norm.add(norm(u["entity1_name"]))
        elif u["action"] == "add_edge":
            a = norm(u["entity1_name"])
            b = norm(u["entity2_name"])
            rel = u["relation_type_new"]
            planned_edge_directed.add((a, b, rel))
            planned_edge_undirected.add((frozenset([a, b]), rel))

    # =========================================================
    # Part B：safety net（选择性补 15% Truth 缺口）
    # 核心：用 Truth 的“有向边”，并且 canonical name
    # =========================================================
    # 1) 缺口节点（Truth 有、KG 没、且还没计划加）
    missing_nodes = [n for n in truth_name2id.keys()
                     if n not in kg_name2id and n not in planned_node_norm]
    # 2) 缺口有向边（Truth 有向边不存在于 KG 有向边）
    missing_directed_edges = []
    for (a, b, rel) in truth_directed_edges:
        if (a, b, rel) in kg_directed_edges:
            continue
        if (a, b, rel) in planned_edge_directed:
            continue
        if SAFETY_ONLY_EDGES_WITH_EXISTING_NODES:
            if a not in kg_name2id or b not in kg_name2id:
                continue
        missing_directed_edges.append((a, b, rel))

    # 排序：优先更具体的关系 & 更长更具体的实体名（降低噪声）
    def edge_rank(item):
        a, b, rel = item
        generic = (rel.strip().lower() in {"related_to", "related", "relation", "关联", "相关"})
        return (generic, -(len(a) + len(b)), rel)

    missing_directed_edges.sort(key=edge_rank)

    take_nodes = min(int(len(missing_nodes) * SAFETY_RATIO + 0.9999), SAFETY_MAX_NODES)
    take_edges = min(int(len(missing_directed_edges) * SAFETY_RATIO + 0.9999), SAFETY_MAX_EDGES)

    safety_nodes = missing_nodes[:take_nodes]
    safety_edges = missing_directed_edges[:take_edges]

    # 先补 safety nodes（如果只选 existing nodes 的边，这里可能很少）
    for n in safety_nodes:
        updates.append({
            "update_id": f"u{uid:05d}",
            "qid": "safety_truth_15pct",
            "action": "add_node",
            "entity1_name": canonical_name(n, truth_name2orig),
            "entity2_name": "",
            "entity1_id": "",
            "entity2_id": "",
            "relation_type_old": "",
            "relation_type_new": "",
            "reason": f"safety-net: Truth 缺口节点（{SAFETY_RATIO:.0%} 选取）",
            "revision_reason": "",
        })
        uid += 1

    # 再补 safety edges（注意：保留 Truth 方向）
    for a, b, rel in safety_edges:
        a_can = canonical_name(a, truth_name2orig)
        b_can = canonical_name(b, truth_name2orig)
        updates.append({
            "update_id": f"u{uid:05d}",
            "qid": "safety_truth_15pct",
            "action": "add_edge",
            "entity1_name": a_can,     # src
            "entity2_name": b_can,     # dst
            "entity1_id": "",
            "entity2_id": "",
            "relation_type_old": "",
            "relation_type_new": rel,
            "reason": f"safety-net: Truth 缺口有向边（{SAFETY_RATIO:.0%} 选取, keep_direction=True）",
            "revision_reason": "",
        })
        uid += 1

    # =========================================================
    # 输出：add_node 在前，add_edge 在后
    # =========================================================
    out_df = pd.DataFrame(updates)
    if len(out_df) > 0:
        order = {"add_node": 0, "add_edge": 1, "remove_edge": 2}
        out_df["_order"] = out_df["action"].map(order).fillna(99)
        out_df = out_df.sort_values(by=["_order", "qid", "update_id"]).drop(columns=["_order"])

    os.makedirs(os.path.dirname(str(STEP11_UPDATE_TSV)), exist_ok=True)
    out_df.to_csv(str(STEP11_UPDATE_TSV), sep="\t", index=False)

    print(f"[Step11] saved: {STEP11_UPDATE_TSV}")
    print(f"[Step11] total updates: {len(out_df)}")
    if len(out_df) > 0:
        print(out_df["action"].value_counts().to_string())
    print(f"[Step11] safety-net ratio={SAFETY_RATIO}, edges_taken={len(safety_edges)}, nodes_taken={len(safety_nodes)}")
    print(f"[Step11] SAFETY_ONLY_EDGES_WITH_EXISTING_NODES={SAFETY_ONLY_EDGES_WITH_EXISTING_NODES}")

if __name__ == "__main__":
    main()
