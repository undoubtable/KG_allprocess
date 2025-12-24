# Step11_generate_kg_update_suggestions.py

import pandas as pd
import re
from pathlib import Path
from pipeline_config import (
    STEP4_NODES_TSV,
    STEP4_EDGES_TSV,
    STEP10_Q_REVISED_TSV,
    STEP11_UPDATE_TSV,
)

# =========================
# 路径（按你的要求：仍用 STEP4_*）
# =========================
NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)
REVISED_Q_TSV = str(STEP10_Q_REVISED_TSV)
OUTPUT_TSV = str(STEP11_UPDATE_TSV)


# =========================
# 工具函数
# =========================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_entities(text: str, name2id: dict):
    """
    从文本中找 KG 实体（基于 name 子串匹配）
    返回 set(node_id)
    """
    found = set()
    text_norm = normalize(text)
    for name, nid in name2id.items():
        if name in text_norm:
            found.add(nid)
    return found


def has_edge(edges_df, src, dst):
    return (
        ((edges_df["src_id"] == src) & (edges_df["dst_id"] == dst)).any()
        or ((edges_df["src_id"] == dst) & (edges_df["dst_id"] == src)).any()
    )


# =========================
# 主流程
# =========================
def main():
    # 1. 读 KG
    nodes_df = pd.read_csv(NODES_TSV, sep="\t")
    edges_df = pd.read_csv(EDGES_TSV, sep="\t")

    # name -> node_id（统一 lower）
    name2id = {
        normalize(row["name"]): row["node_id"]
        for _, row in nodes_df.iterrows()
    }

    # node_id -> name（回写用）
    id2name = {
        row["node_id"]: row["name"]
        for _, row in nodes_df.iterrows()
    }

    # 2. 读 Step10 改写后的题目
    q_df = pd.read_csv(REVISED_Q_TSV, sep="\t")

    required_cols = {
        "qid",
        "original_question",
        "revised_question",
        "changed",
        "revision_reason",
    }
    missing = required_cols - set(q_df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    suggestions = []

    # 3. 逐题生成 KG 更新建议
    for _, row in q_df.iterrows():
        if str(row["changed"]).lower() != "yes":
            continue

        qid = row["qid"]
        orig_q = str(row["original_question"])
        rev_q = str(row["revised_question"])
        feedback = str(row["revision_reason"])

        orig_entities = extract_entities(orig_q, name2id)
        rev_entities = extract_entities(rev_q, name2id)

        # 新出现的实体
        added_entities = rev_entities - orig_entities

        # 3.1 新实体 → add_node
        for nid in added_entities:
            suggestions.append(
                {
                    "qid": qid,
                    "action": "add_node",
                    "entity1_id": nid,
                    "entity1_name": id2name.get(nid, ""),
                    "entity2_id": "",
                    "entity2_name": "",
                    "relation_type_old": "",
                    "relation_type_new": "",
                    "revision_reason": feedback,
                }
            )

        # 3.2 实体组合但 KG 中无边 → add_edge
        rev_entity_list = list(rev_entities)
        for i in range(len(rev_entity_list)):
            for j in range(i + 1, len(rev_entity_list)):
                src = rev_entity_list[i]
                dst = rev_entity_list[j]
                if not has_edge(edges_df, src, dst):
                    suggestions.append(
                        {
                            "qid": qid,
                            "action": "add_edge",
                            "entity1_id": src,
                            "entity1_name": id2name.get(src, ""),
                            "entity2_id": dst,
                            "entity2_name": id2name.get(dst, ""),
                            "relation_type_old": "",
                            "relation_type_new": "related_to",
                            "revision_reason": feedback,
                        }
                    )

    # 4. 输出
    out_df = pd.DataFrame(suggestions)
    out_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"[Step11] KG update suggestions saved to {OUTPUT_TSV}")
    print(f"[Step11] Total suggestions: {len(out_df)}")


if __name__ == "__main__":
    main()
