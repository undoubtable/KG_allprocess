import csv
import os
import re
from pipeline_config import  STEP4_EDGES_TSV, STEP4_NODES_TSV, STEP10_Q_REVISED_TSV, STEP11_UPDATE_TSV 

# ======== 配置：改成你自己的路径 ========
# NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
# EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"
NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)
# 这里要填【已经修正过问题】的 TSV
# 如果你现在用的是 Step10_edit_questions.py，那就保持下面这样；
# 如果你改用 Step10_auto_improve_mcq.py，就改成对应的输出路径。
# Q_REVISED_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step10_output\第一讲_MCQ_auto_revised.tsv"
Q_REVISED_TSV = str(STEP10_Q_REVISED_TSV)
# OUTPUT_UPDATE_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step11_output\第一讲_KG_update_suggestions.tsv"
OUTPUT_UPDATE_TSV = str(STEP11_UPDATE_TSV )

# ============ 工具函数 ============

def load_nodes(path):
    """读取节点表：返回 name -> [node_id, ...] 映射 和 node_id -> name"""
    name2ids = {}
    id2name = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            nid = row["node_id"]
            name = row["name"].strip()
            id2name[nid] = name
            if name:
                name2ids.setdefault(name, []).append(nid)
    return name2ids, id2name


def load_edges(path):
    """读取边表"""
    edges = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            edges.append(row)
    return edges


def load_revised_questions(path):
    """
    读取 Step10 的修改后问题文件：
    需要包含字段至少：
      - qid
      - original_question
      - revised_question
      - changed (yes/no)
      - revision_reason（可选）
    """
    qs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qs.append(row)
    return qs


def find_entities_from_text(q_text: str, node_names):
    """
    不再依赖「……」，而是：
    在题干文本中查找出现的 KG 节点名字，返回一个 set(name)。

    简单做法：字符串包含匹配。
    """
    q_text = q_text.strip()
    ents = set()
    if not q_text:
        return ents
    for name in node_names:
        if name and name in q_text:
            ents.add(name)
    return ents


# ============ 核心逻辑 ============

def generate_update_suggestions(nodes_name2ids, id2name, edges, revised_qs):
    """
    根据修改后的问题生成 KG 更新建议。
    返回 suggestions 列表。
    """
    # 建 (name1, name2) -> 边 列表 的索引，方便判断两个名字之间是否已有边
    pair2edges = {}
    for e in edges:
        s_id = e["src_id"]
        t_id = e["dst_id"]
        s_name = id2name.get(s_id)
        t_name = id2name.get(t_id)
        if not s_name or not t_name:
            continue
        key = (s_name, t_name)
        pair2edges.setdefault(key, []).append(e)
        # 视关系为无向，用反向也建索引
        key_rev = (t_name, s_name)
        pair2edges.setdefault(key_rev, []).append(e)

    suggestions = []
    update_id = 1
    all_node_names = list(nodes_name2ids.keys())

    for q in revised_qs:
        qid = q.get("qid", "")
        orig_q = q.get("original_question", "")
        rev_q = q.get("revised_question", "")
        changed = (q.get("changed", "") or "").strip()
        revision_reason = q.get("revision_reason", "").strip()

        # 只分析确实改动过的题
        if changed.lower() != "yes":
            continue

        # ===== 关键改动：用 KG 节点名字来做实体识别 =====
        orig_ents = find_entities_from_text(orig_q, all_node_names)
        rev_ents = find_entities_from_text(rev_q, all_node_names)

        # 调试输出（你可以先看终端确认是否识别到实体）
        print(f"QID={qid}")
        print(f"  原题实体: {orig_ents}")
        print(f"  新题实体: {rev_ents}")

        if not orig_ents and not rev_ents:
            # 两个问题里都没有匹配到任何 KG 节点名，那就没法推 KG 更新
            continue

        removed_ents = orig_ents - rev_ents
        added_ents = rev_ents - orig_ents
        kept_ents = orig_ents & rev_ents

        # === 1) 被移除的实体：考虑 remove_edge ===
        for ent in removed_ents:
            if ent not in nodes_name2ids:
                continue  # KG 里本来就没这个实体

            node_ids = nodes_name2ids[ent]
            for other in kept_ents:
                if other not in nodes_name2ids:
                    continue
                other_ids = nodes_name2ids[other]

                for nid in node_ids:
                    for oid in other_ids:
                        # 看 (ent, other) 是否有边
                        for e in pair2edges.get((ent, other), []):
                            if (
                                (e["src_id"] == nid and e["dst_id"] == oid) or
                                (e["src_id"] == oid and e["dst_id"] == nid)
                            ):
                                suggestions.append({
                                    "update_id": f"u{update_id:05d}",
                                    "qid": qid,
                                    "action": "remove_edge",
                                    "entity1_name": ent,
                                    "entity2_name": other,
                                    "entity1_id": nid,
                                    "entity2_id": oid,
                                    "relation_type_old": e["relation_type"],
                                    "relation_type_new": "",
                                    "reason": "该实体在修改后的问题中被移除，可能不应与保留实体维持此关系",
                                    "revision_reason": revision_reason,
                                })
                                update_id += 1

        # === 2) 新增的实体：考虑 add_node / add_edge ===
        for ent in added_ents:
            if ent not in nodes_name2ids:
                # KG 中不存在，建议新增节点
                suggestions.append({
                    "update_id": f"u{update_id:05d}",
                    "qid": qid,
                    "action": "add_node",
                    "entity1_name": ent,
                    "entity2_name": "",
                    "entity1_id": "",
                    "entity2_id": "",
                    "relation_type_old": "",
                    "relation_type_new": "",
                    "reason": "修改后的问题新增实体，KG 中不存在该实体，建议添加节点",
                    "revision_reason": revision_reason,
                })
                update_id += 1
            else:
                # 实体已存在，检查与 kept_ents 是否已有边；没有则建议加边
                for other in kept_ents:
                    if other not in nodes_name2ids:
                        continue

                    has_edge = False
                    for nid in nodes_name2ids[ent]:
                        for oid in nodes_name2ids[other]:
                            for e in pair2edges.get((ent, other), []):
                                if (
                                    (e["src_id"] == nid and e["dst_id"] == oid) or
                                    (e["src_id"] == oid and e["dst_id"] == nid)
                                ):
                                    has_edge = True
                                    break
                            if has_edge:
                                break
                        if has_edge:
                            break

                    if not has_edge:
                        suggestions.append({
                            "update_id": f"u{update_id:05d}",
                            "qid": qid,
                            "action": "add_edge",
                            "entity1_name": ent,
                            "entity2_name": other,
                            "entity1_id": ",".join(nodes_name2ids[ent]),
                            "entity2_id": ",".join(nodes_name2ids[other]),
                            "relation_type_old": "",
                            "relation_type_new": "related_to",
                            "reason": "修改后的问题引入了新的实体组合，但 KG 中两者没有关系，建议添加边",
                            "revision_reason": revision_reason,
                        })
                        update_id += 1

    return suggestions


def save_suggestions(suggestions, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "update_id",
        "qid",
        "action",
        "entity1_name",
        "entity2_name",
        "entity1_id",
        "entity2_id",
        "relation_type_old",
        "relation_type_new",
        "reason",
        "revision_reason",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for s in suggestions:
            writer.writerow(s)

    print(f"✅ 已生成 KG 更新建议：{out_path}")
    print(f"   共 {len(suggestions)} 条。")


def main():
    if not os.path.exists(NODES_TSV):
        raise FileNotFoundError(NODES_TSV)
    if not os.path.exists(EDGES_TSV):
        raise FileNotFoundError(EDGES_TSV)
    if not os.path.exists(Q_REVISED_TSV):
        raise FileNotFoundError(Q_REVISED_TSV)

    nodes_name2ids, id2name = load_nodes(NODES_TSV)
    edges = load_edges(EDGES_TSV)
    revised_qs = load_revised_questions(Q_REVISED_TSV)

    suggestions = generate_update_suggestions(nodes_name2ids, id2name, edges, revised_qs)
    save_suggestions(suggestions, OUTPUT_UPDATE_TSV)


if __name__ == "__main__":
    main()
