"""
Step12 — 根据 Step11 的更新建议，真正修改知识图谱（KG）

输入：
- 原始 KG 节点文件（STEP4_NODES_TSV）
- 原始 KG 边文件（STEP4_EDGES_TSV）
- Step11 输出的更新建议（STEP11_UPDATE_TSV）

输出：
- 更新后的节点文件（STEP12_NODES_TSV）
- 更新后的边文件（STEP12_EDGES_TSV）

支持的 action:
- add_node    : 新增节点（按 name 去重）
- add_edge    : 新增边（按 (src_id, dst_id, relation_type) 去重，含反向）
- remove_edge : 删除边（根据 entity1_id, entity2_id, relation_type_old 匹配）

Tips:
- AUTO_APPLY = True  → 直接修改 KG
- AUTO_APPLY = False → 只打印建议，不写回文件
"""

import csv
import os
import re
from collections import Counter
from pipeline_config import STEP4_EDGES_TSV, STEP4_NODES_TSV, STEP11_UPDATE_TSV
from pipeline_config import STEP12_EDGES_TSV, STEP12_NODES_TSV

# 原始 KG 节点 & 边（按你要求：仍从 STEP4_* 来）
NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)

# Step11 的更新建议文件
UPDATES_TSV = str(STEP11_UPDATE_TSV)

# 更新后的输出路径
OUTPUT_NODES_TSV = str(STEP12_NODES_TSV)
OUTPUT_EDGES_TSV = str(STEP12_EDGES_TSV)

# True：真的改；False：只打印不改（调试用）
AUTO_APPLY = True


# ============ 加载原始 KG ============

def load_nodes(path):
    nodes = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nodes.append(row)
    return nodes


def load_edges(path):
    edges = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            edges.append(row)
    return edges


def load_updates(path):
    updates = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            updates.append(row)
    return updates


# ============ 工具函数 ============

def norm_name(s: str) -> str:
    """节点名去重用：忽略大小写，合并空白"""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def infer_id_prefix(ids, default_prefix: str):
    """
    从现有 id 推断前缀（最常见的首字母），失败则用 default_prefix
    例如：n00001 -> 'n'
    """
    prefixes = []
    for _id in ids:
        if not _id:
            continue
        _id = str(_id).strip()
        if len(_id) >= 2 and _id[0].isalpha():
            prefixes.append(_id[0])
    if not prefixes:
        return default_prefix
    return Counter(prefixes).most_common(1)[0][0]


def make_next_id(existing_ids, prefix: str):
    """
    给定已有 ID 列表 + 前缀，生成自增 ID：
    n00001, n00002 -> n00003
    """
    max_num = 0
    for _id in existing_ids:
        if not _id:
            continue
        _id = str(_id).strip()
        if len(_id) > 1 and _id[0] == prefix and _id[1:].isdigit():
            max_num = max(max_num, int(_id[1:]))

    def _next():
        nonlocal max_num
        max_num += 1
        return f"{prefix}{max_num:05d}"

    return _next


def build_name_index(nodes):
    """
    name_norm -> [node_row, ...]
    """
    name2nodes = {}
    for n in nodes:
        name = n.get("name", "")
        k = norm_name(name)
        if k:
            name2nodes.setdefault(k, []).append(n)
    return name2nodes


def build_edge_index(edges):
    """(src_id, dst_id, relation_type) -> [edge_row, ...]"""
    index = {}
    for e in edges:
        key = (e["src_id"], e["dst_id"], e["relation_type"])
        index.setdefault(key, []).append(e)
    return index


# ============ 应用更新 ============

def apply_updates(nodes, edges, updates):
    name2nodes = build_name_index(nodes)
    edge_index = build_edge_index(edges)

    # 1) 推断 node/edge 的前缀体系（默认 n/e）
    node_prefix = infer_id_prefix([n.get("node_id", "") for n in nodes], default_prefix="n")
    edge_prefix = infer_id_prefix([e.get("edge_id", "") for e in edges], default_prefix="e")

    # 2) 建立自增 id 生成器
    get_next_node_id = make_next_id([n.get("node_id", "") for n in nodes], prefix=node_prefix)
    get_next_edge_id = make_next_id([e.get("edge_id", "") for e in edges], prefix=edge_prefix)

    added_nodes = 0
    added_edges = 0
    removed_edges = 0

    for u in updates:
        action = (u.get("action") or "").strip()
        e1_name = (u.get("entity1_name") or "").strip()
        e2_name = (u.get("entity2_name") or "").strip()
        e1_id_str = (u.get("entity1_id") or "").strip()
        e2_id_str = (u.get("entity2_id") or "").strip()
        rel_old = (u.get("relation_type_old") or "").strip()
        rel_new = (u.get("relation_type_new") or "").strip() or "related_to"

        update_id = u.get("update_id", "")
        qid = u.get("qid", "")

        print(f"\n处理更新 {update_id} (action={action}, qid={qid})")

        if not AUTO_APPLY:
            print("  [预览模式] AUTO_APPLY=False，不会真正修改 KG。")

        # ---- add_node：新增节点 ----
        if action == "add_node":
            k = norm_name(e1_name)
            if not k:
                print("  [跳过] add_node 缺少 entity1_name")
                continue

            if k in name2nodes:
                print(f"  [跳过] 节点已存在：{e1_name} -> {[n['node_id'] for n in name2nodes[k]]}")
                continue

            if not AUTO_APPLY:
                print(f"  [建议] 添加新节点：name={e1_name}")
                continue

            new_id = get_next_node_id()
            new_node = {
                "node_id": new_id,
                "label": "Concept",
                "name": e1_name,
                "page_no": "-1",
                "sentence_id": "",
            }
            nodes.append(new_node)
            name2nodes.setdefault(k, []).append(new_node)
            added_nodes += 1
            print(f"  [+] 已添加节点：{new_id} ({e1_name})")

        # ---- add_edge：新增边 ----
        elif action == "add_edge":
            # 优先使用建议里的 entity1_id / entity2_id；没有的话就按 name 找
            if e1_id_str:
                e1_ids = [x.strip() for x in e1_id_str.split(",") if x.strip()]
            else:
                e1_ids = [n["node_id"] for n in name2nodes.get(norm_name(e1_name), [])]

            if e2_id_str:
                e2_ids = [x.strip() for x in e2_id_str.split(",") if x.strip()]
            else:
                e2_ids = [n["node_id"] for n in name2nodes.get(norm_name(e2_name), [])]

            if not e1_ids or not e2_ids:
                print(f"  [警告] add_edge 找不到节点ID: {e1_name}({e1_ids}), {e2_name}({e2_ids})")
                continue

            for sid in e1_ids:
                for tid in e2_ids:
                    key1 = (sid, tid, rel_new)
                    key2 = (tid, sid, rel_new)

                    if key1 in edge_index or key2 in edge_index:
                        print(f"  [跳过] 边已存在: {sid} -[{rel_new}]-> {tid}")
                        continue

                    if not AUTO_APPLY:
                        print(f"  [建议] 添加边: {sid} -[{rel_new}]-> {tid}")
                        continue

                    new_eid = get_next_edge_id()
                    new_edge = {
                        "edge_id": new_eid,
                        "src_id": sid,
                        "dst_id": tid,
                        "relation_type": rel_new,
                        "confidence": "0.5",
                        "page_no": "-1",
                        "sentence_id": "",
                    }
                    edges.append(new_edge)
                    edge_index.setdefault((sid, tid, rel_new), []).append(new_edge)
                    added_edges += 1
                    print(f"  [+] 已添加边: {sid} -[{rel_new}]-> {tid} (edge_id={new_eid})")

        # ---- remove_edge：删除边 ----
        elif action == "remove_edge":
            sid = e1_id_str
            tid = e2_id_str
            if not sid or not tid:
                print(f"  [警告] remove_edge 缺少 entity1_id / entity2_id: {sid}, {tid}")
                continue

            rels_to_check = [rel_old] if rel_old else None

            to_remove = []
            for e in edges:
                if ((e["src_id"] == sid and e["dst_id"] == tid) or
                    (e["src_id"] == tid and e["dst_id"] == sid)):
                    if rels_to_check is None or e["relation_type"] in rels_to_check:
                        to_remove.append(e)

            if not to_remove:
                print(f"  [提示] 找不到需要删除的边: {sid} <-> {tid}, rel_old={rel_old}")
                continue

            if not AUTO_APPLY:
                for e in to_remove:
                    print(f"  [建议] 删除边: {e['edge_id']} {e['src_id']} -[{e['relation_type']}]-> {e['dst_id']}")
                continue

            for e in to_remove:
                edges.remove(e)
                removed_edges += 1
                print(f"  [-] 已删除边: {e['edge_id']} {e['src_id']} -[{e['relation_type']}]-> {e['dst_id']}")

        else:
            print(f"  [跳过] 未知 action: {action}")

    print("\n===== 更新统计 =====")
    print(f"  新增节点数：{added_nodes}")
    print(f"  新增边数：{added_edges}")
    print(f"  删除边数：{removed_edges}")

    return nodes, edges


# ============ 保存新的 KG ============

def save_nodes(nodes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["node_id", "label", "name", "page_no", "sentence_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for n in nodes:
            writer.writerow(n)
    print(f"✅ 已保存更新后的节点文件：{path}")


def save_edges(edges, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["edge_id", "src_id", "dst_id", "relation_type", "confidence", "page_no", "sentence_id"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for e in edges:
            writer.writerow(e)
    print(f"✅ 已保存更新后的边文件：{path}")


def main():
    if not os.path.exists(NODES_TSV):
        raise FileNotFoundError(NODES_TSV)
    if not os.path.exists(EDGES_TSV):
        raise FileNotFoundError(EDGES_TSV)
    if not os.path.exists(UPDATES_TSV):
        raise FileNotFoundError(UPDATES_TSV)

    nodes = load_nodes(NODES_TSV)
    edges = load_edges(EDGES_TSV)
    updates = load_updates(UPDATES_TSV)

    print(f"原始节点数：{len(nodes)}，原始边数：{len(edges)}")
    print(f"更新建议数：{len(updates)}，AUTO_APPLY = {AUTO_APPLY}")

    new_nodes, new_edges = apply_updates(nodes, edges, updates)

    save_nodes(new_nodes, OUTPUT_NODES_TSV)
    save_edges(new_edges, OUTPUT_EDGES_TSV)


if __name__ == "__main__":
    main()
