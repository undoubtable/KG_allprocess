import os
import csv
from pipeline_config import  STEP4_NODES_TSV, STEP4_EDGES_TSV, STEP5_GLOBAL_NODES, STEP5_GLOBAL_EDGES

out_nodes_path = str(STEP4_NODES_TSV)
out_edges_path = str(STEP4_EDGES_TSV)
# ========== 配置区域：改成你自己的路径 ==========
# 如果之后有“第二讲、第三讲…”，就在这个 list 里继续加元素
input_kgs = [
    {
        "name": "第一讲",
        "nodes_path": str(STEP4_NODES_TSV),
        "edges_path": str(STEP4_EDGES_TSV),
    },
    # 未来可以这样加：
    # {
    #     "name": "第二讲",
    #     "nodes_path": r"...\第二讲_KG_nodes.tsv",
    #     "edges_path": r"...\第二讲_KG_edges.tsv",
    # },
]

# 全局 KG 的输出路径
global_nodes_path = str(STEP5_GLOBAL_NODES)
global_edges_path = str(STEP5_GLOBAL_EDGES)
# ===============================================


def load_nodes(path, kg_name):
    nodes = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["_kg_name"] = kg_name  # 记录来自哪个讲义
            nodes.append(row)
    return nodes


def load_edges(path, kg_name):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["_kg_name"] = kg_name
            edges.append(row)
    return edges


def build_global_nodes(all_nodes):
    """
    按 (name, label) 合并多个讲义里的节点，构造全局唯一节点：
    - 同一个 (name, label) -> 一个 global_node_id (g00001...)
    - 返回:
        global_nodes: 去重后的节点列表
        local2global: (kg_name, local_node_id) -> global_node_id 的映射
    """
    key2global_id = {}
    local2global = {}
    global_nodes = []
    idx = 1

    for n in all_nodes:
        name = n.get("name", "")
        label = n.get("label", "Entity")
        kg_name = n["_kg_name"]
        local_id = n["node_id"]

        key = (name, label)

        if key not in key2global_id:
            gid = f"g{idx:05d}"
            key2global_id[key] = gid
            idx += 1

            # 记录第一次出现的位置，方便追踪（可选）
            global_nodes.append(
                {
                    "global_node_id": gid,
                    "name": name,
                    "label": label,
                    "example_page_no": n.get("page_no", ""),
                    "example_sentence_id": n.get("sentence_id", ""),
                    "example_kg": kg_name,
                }
            )

        local2global[(kg_name, local_id)] = key2global_id[key]

    return global_nodes, local2global


def build_global_edges(all_edges, local2global):
    """
    把各讲义里的边映射到全局节点上：
    - 源/目标 local node_id -> global_node_id
    - 同一条 (src, dst, relation_type) 可以考虑去重，这里保留置信度最高的那条
    """
    temp = {}  # key -> edge dict
    for e in all_edges:
        kg_name = e["_kg_name"]
        local_src = e["src_id"]
        local_dst = e["dst_id"]

        key_src = (kg_name, local_src)
        key_dst = (kg_name, local_dst)

        if key_src not in local2global or key_dst not in local2global:
            continue

        src = local2global[key_src]
        dst = local2global[key_dst]

        # 不连接自己
        if src == dst:
            continue

        rel_type = e.get("relation_type", "related_to")

        try:
            conf = float(e.get("confidence", 0.0))
        except ValueError:
            conf = 0.0

        # 用 (src, dst, relation_type) 去重，多次出现保留置信度最高的一条
        key = (src, dst, rel_type)
        if key not in temp or conf > temp[key]["confidence"]:
            temp[key] = {
                "src_id": src,
                "dst_id": dst,
                "relation_type": rel_type,
                "confidence": conf,
                # 下面这些只是为了方便回溯
                "example_page_no": e.get("page_no", ""),
                "example_sentence_id": e.get("sentence_id", ""),
                "example_kg": kg_name,
            }

    # 把字典转成列表，并给 edge 编号
    global_edges = []
    idx = 1
    for _, v in temp.items():
        global_edges.append(
            {
                "global_edge_id": f"ge{idx:05d}",
                "src_id": v["src_id"],
                "dst_id": v["dst_id"],
                "relation_type": v["relation_type"],
                "confidence": v["confidence"],
                "example_page_no": v["example_page_no"],
                "example_sentence_id": v["example_sentence_id"],
                "example_kg": v["example_kg"],
            }
        )
        idx += 1

    return global_edges


def save_tsv(rows, fieldnames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"✅ 已保存：{path}（{len(rows)} 行）")


def main():
    # 1）读入所有 Step4 KG
    all_nodes = []
    all_edges = []
    for cfg in input_kgs:
        kg_name = cfg["name"]
        print(f"\n📚 读取讲义：{kg_name}")
        nodes = load_nodes(cfg["nodes_path"], kg_name)
        edges = load_edges(cfg["edges_path"], kg_name)
        print(f"   节点数：{len(nodes)}，边数：{len(edges)}")
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    print(f"\n📊 合计：节点 {len(all_nodes)} 个（有重复），边 {len(all_edges)} 条（有重复）")

    # 2）构建全局唯一节点
    global_nodes, local2global = build_global_nodes(all_nodes)
    print(f"✨ 全局唯一实体（节点）数量：{len(global_nodes)}")

    # 3）映射边到全局节点，并做简单去重
    global_edges = build_global_edges(all_edges, local2global)
    print(f"✨ 全局边数量（去重后）：{len(global_edges)}")

    # 4）保存
    save_tsv(
        global_nodes,
        [
            "global_node_id",
            "name",
            "label",
            "example_page_no",
            "example_sentence_id",
            "example_kg",
        ],
        global_nodes_path,
    )

    save_tsv(
        global_edges,
        [
            "global_edge_id",
            "src_id",
            "dst_id",
            "relation_type",
            "confidence",
            "example_page_no",
            "example_sentence_id",
            "example_kg",
        ],
        global_edges_path,
    )

    # 简单预览
    print("\n📌 节点示例：")
    for n in global_nodes[:10]:
        print(n)

    print("\n📌 边示例：")
    for e in global_edges[:10]:
        print(e)


if __name__ == "__main__":
    main()
