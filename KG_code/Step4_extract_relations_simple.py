import os
import csv

# ========= 配置区域：改成你自己的路径 =========
# Step3 输出的“实体列表”
entity_tsv_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step3_output\第一讲_实体列表.tsv"

# 输出：唯一节点 & 关系边
out_nodes_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
out_edges_path = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"


def load_entities(tsv_path):
    """
    读取 Step3 的实体 TSV：
    entity_id, sentence_id, page_no, mention, start_char, end_char, ent_type, confidence
    """
    entities = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # 做一下类型转换，方便后面排序
            row["page_no"] = int(row.get("page_no", 0))
            row["start_char"] = int(row.get("start_char", 0))
            row["end_char"] = int(row.get("end_char", 0))
            try:
                row["confidence"] = float(row.get("confidence", 0.0))
            except ValueError:
                row["confidence"] = 0.0
            entities.append(row)
    return entities


def build_unique_nodes(entities):
    """
    按 (mention, ent_type) 合并实体，生成唯一节点：
    - 同名 + 同类型 → 一个 node_id（n00001, n00002, ...）
    - 顺便生成映射： (mention, ent_type) -> node_id
    """
    key2node_id = {}
    nodes = []
    idx = 1

    for e in entities:
        mention = e["mention"]
        ent_type = e.get("ent_type", "Entity")
        key = (mention, ent_type)

        if key not in key2node_id:
            node_id = f"n{idx:05d}"
            key2node_id[key] = node_id
            idx += 1

            # 记录第一次出现时的一些位置信息（只是为了方便查看）
            nodes.append(
                {
                    "node_id": node_id,
                    "name": mention,
                    "label": ent_type,
                    "page_no": e.get("page_no", ""),
                    "sentence_id": e.get("sentence_id", ""),
                }
            )

    return nodes, key2node_id


def build_edges_by_sentence(entities, key2node_id):
    """
    在“句子内部”按出现顺序连接相邻实体：
    - 先按 sentence_id 分组
    - 同一句内按 start_char 排序
    - 相邻两个实体 -> 连一条边
    - 边指向的是“合并后的 node_id”
    """
    # 按句子分组
    ents_by_sent = {}
    for e in entities:
        sid = e["sentence_id"]
        ents_by_sent.setdefault(sid, []).append(e)

    edges = []
    edge_idx = 1

    for sent_id, ents in ents_by_sent.items():
        # 按出现顺序排序
        ents_sorted = sorted(ents, key=lambda x: x["start_char"])
        if len(ents_sorted) < 2:
            continue

        page_no = ents_sorted[0]["page_no"]

        for i in range(len(ents_sorted) - 1):
            h = ents_sorted[i]
            t = ents_sorted[i + 1]

            h_key = (h["mention"], h.get("ent_type", "Entity"))
            t_key = (t["mention"], t.get("ent_type", "Entity"))

            # 找到对应的唯一 node_id
            src = key2node_id.get(h_key)
            dst = key2node_id.get(t_key)
            if not src or not dst:
                continue

            # 如果合并后变成同一个节点，就没必要连自己
            if src == dst:
                continue

            edges.append(
                {
                    "edge_id": f"e{edge_idx:05d}",
                    "src_id": src,
                    "dst_id": dst,
                    "relation_type": "related_to",  # 先统一叫 related_to，之后可以细分
                    "page_no": page_no,
                    "sentence_id": sent_id,
                    "confidence": min(h["confidence"], t["confidence"]),  # 简单给个置信度
                }
            )
            edge_idx += 1

    return edges


def save_tsv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"✅ 已保存：{path}（{len(rows)} 行）")


def main():
    # 1) 读取实体
    entities = load_entities(entity_tsv_path)
    print(f"📄 实体总数（包括重复提及）：{len(entities)}")

    # 2) 合并成唯一节点
    nodes, key2node_id = build_unique_nodes(entities)
    print(f"✨ 合并后唯一实体（节点）数量：{len(nodes)}")

    # 3) 在句子内按顺序连边（使用合并后的节点）
    edges = build_edges_by_sentence(entities, key2node_id)
    print(f"🔗 生成边数量：{len(edges)}")

    # 4) 保存
    save_tsv(
        nodes,
        ["node_id", "name", "label", "page_no", "sentence_id"],
        out_nodes_path,
    )
    save_tsv(
        edges,
        ["edge_id", "src_id", "dst_id", "relation_type", "page_no", "sentence_id", "confidence"],
        out_edges_path,
    )


if __name__ == "__main__":
    main()
