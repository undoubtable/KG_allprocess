import csv
import os
import re
from collections import Counter

from pipeline_config import (
    STEP4_NODES_TSV,     # ✅ 被更新的 KG-version1（如果你实际想更新 Step45，就把这里换成 STEP45）
    STEP4_EDGES_TSV,
    STEP11_UPDATE_TSV,
    STEP12_NODES_TSV,
    STEP12_EDGES_TSV,
)

# =========================
# 路径
# =========================
NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)
UPDATES_TSV = str(STEP11_UPDATE_TSV)

OUTPUT_NODES_TSV = str(STEP12_NODES_TSV)
OUTPUT_EDGES_TSV = str(STEP12_EDGES_TSV)

AUTO_APPLY = True  # True: 写回；False: 只打印预览

# =========================
# 工具函数
# =========================
def norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def infer_id_prefix(ids, default_prefix: str):
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

def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows

def save_tsv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def build_name_index(nodes):
    idx = {}
    for n in nodes:
        k = norm_name(n.get("name", ""))
        if k:
            idx.setdefault(k, []).append(n)
    return idx

def build_edge_index(edges):
    idx = {}
    for e in edges:
        key = (e["src_id"], e["dst_id"], e["relation_type"])
        idx.setdefault(key, []).append(e)
    return idx

def edge_exists_undirected(edge_index, src, dst, rel):
    return (src, dst, rel) in edge_index or (dst, src, rel) in edge_index

# =========================
# 主流程：应用更新
# =========================
def main():
    if not os.path.exists(NODES_TSV):
        raise FileNotFoundError(NODES_TSV)
    if not os.path.exists(EDGES_TSV):
        raise FileNotFoundError(EDGES_TSV)
    if not os.path.exists(UPDATES_TSV):
        raise FileNotFoundError(UPDATES_TSV)

    nodes = load_tsv(NODES_TSV)
    edges = load_tsv(EDGES_TSV)
    updates = load_tsv(UPDATES_TSV)

    print(f"原始节点数：{len(nodes)}，原始边数：{len(edges)}")
    print(f"更新建议数：{len(updates)}，AUTO_APPLY = {AUTO_APPLY}")

    # 强制应用顺序：add_node -> add_edge -> remove_edge
    action_order = {"add_node": 0, "add_edge": 1, "remove_edge": 2}
    updates.sort(key=lambda u: (u.get("qid", ""), action_order.get((u.get("action") or "").strip(), 99), u.get("update_id", "")))

    name2nodes = build_name_index(nodes)
    edge_index = build_edge_index(edges)

    node_prefix = infer_id_prefix([n.get("node_id", "") for n in nodes], default_prefix="n")
    edge_prefix = infer_id_prefix([e.get("edge_id", "") for e in edges], default_prefix="e")
    next_node_id = make_next_id([n.get("node_id", "") for n in nodes], prefix=node_prefix)
    next_edge_id = make_next_id([e.get("edge_id", "") for e in edges], prefix=edge_prefix)

    added_nodes = 0
    added_edges = 0
    removed_edges = 0

    for u in updates:
        action = (u.get("action") or "").strip()
        qid = u.get("qid", "")
        update_id = u.get("update_id", "")

        e1_name = (u.get("entity1_name") or "").strip()
        e2_name = (u.get("entity2_name") or "").strip()
        e1_id_str = (u.get("entity1_id") or "").strip()
        e2_id_str = (u.get("entity2_id") or "").strip()

        rel_old = (u.get("relation_type_old") or "").strip()
        rel_new = (u.get("relation_type_new") or "").strip() or "related_to"

        print(f"\n处理更新 {update_id} (action={action}, qid={qid})")

        # ========== add_node ==========
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

            new_id = next_node_id()
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

        # ========== add_edge ==========
        elif action == "add_edge":
            # 优先使用 id；如果为空就按 name 找（推荐：Step11 输出 id 为空）
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
                    if edge_exists_undirected(edge_index, sid, tid, rel_new):
                        print(f"  [跳过] 边已存在: {sid} -[{rel_new}]-> {tid}")
                        continue

                    if not AUTO_APPLY:
                        print(f"  [建议] 添加边: {sid} -[{rel_new}]-> {tid}")
                        continue

                    new_eid = next_edge_id()
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

        # ========== remove_edge ==========
        elif action == "remove_edge":
            sid = e1_id_str
            tid = e2_id_str
            if not sid or not tid:
                print(f"  [警告] remove_edge 缺少 entity1_id / entity2_id: {sid}, {tid}")
                continue

            to_remove = []
            for e in edges:
                if ((e["src_id"] == sid and e["dst_id"] == tid) or
                    (e["src_id"] == tid and e["dst_id"] == sid)):
                    if (not rel_old) or (e["relation_type"] == rel_old):
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

    # 保存
    save_tsv(OUTPUT_NODES_TSV, ["node_id","label","name","page_no","sentence_id"], nodes)
    save_tsv(OUTPUT_EDGES_TSV, ["edge_id","src_id","dst_id","relation_type","confidence","page_no","sentence_id"], edges)

    print(f"✅ 已保存更新后的节点文件：{OUTPUT_NODES_TSV}")
    print(f"✅ 已保存更新后的边文件：{OUTPUT_EDGES_TSV}")

if __name__ == "__main__":
    main()
