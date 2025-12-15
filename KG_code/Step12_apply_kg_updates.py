"""
Step12 — 根据 Step11 的更新建议，真正修改知识图谱（KG）

输入：
- 原始 KG 节点文件：第一讲_KG_nodes.tsv
- 原始 KG 边文件：第一讲_KG_edges.tsv
- Step11 输出的更新建议：第一讲_KG_update_suggestions.tsv

输出：
- 更新后的节点文件：第一讲_KG_nodes_updated.tsv
- 更新后的边文件：第一讲_KG_edges_updated.tsv

支持的 action:
- add_node    : 新增节点
- add_edge    : 新增边
- remove_edge : 删除边（根据 entity1_id, entity2_id, relation_type_old 匹配）

Tips:
- AUTO_APPLY = True  → 直接修改 KG
- AUTO_APPLY = False → 只打印建议，不写回文件
"""

import csv
import os

# ======== 配置：根据你自己的路径调整 ========

# 原始 KG 节点 & 边
NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"

# Step11 的更新建议文件
UPDATES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step11_output\第一讲_KG_update_suggestions.tsv"

# 更新后的输出路径
OUTPUT_NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_nodes_updated.tsv"
OUTPUT_EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_edges_updated.tsv"

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

def make_next_node_id(nodes):
    """
    创建一个生成新 node_id 的函数：
    现有节点形如 e00001, e00002，则下一个从 max+1 开始。
    """
    max_num = 0
    for n in nodes:
        nid = n.get("node_id", "")
        if len(nid) > 1 and nid[1:].isdigit():
            num = int(nid[1:])
            max_num = max(max_num, num)

    def _next():
        nonlocal max_num
        max_num += 1
        return f"e{max_num:05d}"

    return _next


def make_next_edge_id(edges):
    """
    创建一个生成新 edge_id 的函数：
    现有边形如 t00001, t00002，则下一个从 max+1 开始。
    """
    max_num = 0
    for e in edges:
        eid = e.get("edge_id", "")
        if len(eid) > 1 and eid[1:].isdigit():
            num = int(eid[1:])
            max_num = max(max_num, num)

    def _next():
        nonlocal max_num
        max_num += 1
        return f"t{max_num:05d}"

    return _next


def build_name_index(nodes):
    """name -> [node_row, ...]"""
    name2nodes = {}
    for n in nodes:
        name = n.get("name", "").strip()
        if name:
            name2nodes.setdefault(name, []).append(n)
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

    get_next_node_id = make_next_node_id(nodes)
    get_next_edge_id = make_next_edge_id(edges)

    added_nodes = 0
    added_edges = 0
    removed_edges = 0

    for u in updates:
        action = u["action"].strip()
        e1_name = u["entity1_name"].strip()
        e2_name = u["entity2_name"].strip()
        e1_id_str = u["entity1_id"].strip()
        e2_id_str = u["entity2_id"].strip()
        rel_old = u["relation_type_old"].strip()
        rel_new = u["relation_type_new"].strip() or "related_to"

        print(f"\n处理更新 {u['update_id']} (action={action}, qid={u['qid']})")

        if not AUTO_APPLY:
            print("  [预览模式] AUTO_APPLY=False，不会真正修改 KG。")

        # ---- add_node：新增节点 ----
        if action == "add_node":
            if e1_name in name2nodes:
                print(f"  [跳过] 节点「{e1_name}」已存在，node_id = {[n['node_id'] for n in name2nodes[e1_name]]}")
                continue

            if not AUTO_APPLY:
                print(f"  [建议] 添加新节点：name={e1_name}")
                continue

            new_id = get_next_node_id()
            new_node = {
                "node_id": new_id,
                "label": "Concept",      # 你可以按需要设置 label
                "name": e1_name,
                "page_no": "-1",
                "sentence_id": "",
            }
            nodes.append(new_node)
            name2nodes.setdefault(e1_name, []).append(new_node)
            added_nodes += 1
            print(f"  [+] 已添加节点：{new_id} ({e1_name})")

        # ---- add_edge：新增边 ----
        elif action == "add_edge":
            # 优先使用建议里的 entity1_id / entity2_id；没有的话就按 name 找
            if e1_id_str:
                e1_ids = [x.strip() for x in e1_id_str.split(",") if x.strip()]
            else:
                e1_ids = [n["node_id"] for n in name2nodes.get(e1_name, [])]

            if e2_id_str:
                e2_ids = [x.strip() for x in e2_id_str.split(",") if x.strip()]
            else:
                e2_ids = [n["node_id"] for n in name2nodes.get(e2_name, [])]

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
