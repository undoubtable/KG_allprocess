# Step11_KG_update_suggestions.py  (IMPROVED)

import pandas as pd
import re
from pathlib import Path
from pipeline_config import (
    STEP45_NODES_TSV,
    STEP45_EDGES_TSV,
    STEP10_Q_REVISED_TSV,
    STEP11_UPDATE_TSV,
)

# =========================
# 路径（保持你现在的写法）
# =========================
NODES_TSV = str(STEP45_NODES_TSV)
EDGES_TSV = str(STEP45_EDGES_TSV)
REVISED_Q_TSV = str(STEP10_Q_REVISED_TSV)
OUTPUT_TSV = str(STEP11_UPDATE_TSV)

# =========================
# 可调参数（建议先用默认）
# =========================
MAX_EDGES_PER_QID = 2          # 每个题最多建议多少条 add_edge（强烈建议 1~3）
MAX_NODES_PER_QID = 3          # 每个题最多建议多少条 add_node（防止误匹配爆炸）

# 过滤“泛词”，这些不应该被当成 KG 实体
STOPLIST = {
    "因素", "过程", "影响", "特点", "方法", "模型", "系统", "机制", "结果", "作用", "现象",
    "问题", "内容", "信息", "数据", "概念", "原理", "意义", "目的", "步骤", "方式",
    "指标", "能力", "结构", "特征", "性质", "条件", "类型", "阶段", "水平", "方面",
    # 英文泛词
    "factor", "process", "effect", "feature", "method", "model", "system", "mechanism",
    "result", "function", "phenomenon", "problem", "content", "information", "data",
    "concept", "principle", "meaning", "purpose", "step", "way", "metric", "ability",
    "structure", "characteristic", "property", "condition", "type", "stage", "level", "aspect",
}

# “语言润色”反馈关键词：出现这些基本不应改 KG
LANGUAGE_ONLY_PATTERNS = [
    r"更清晰", r"更自然", r"措辞", r"表达", r"语法", r"避免歧义", r"更流畅", r"更准确表述",
    r"用词", r"改写", r"润色", r"调整表述", r"更符合", r"更易理解",
    r"more clear", r"more natural", r"wording", r"grammar", r"avoid ambiguity", r"fluency",
]

# “事实/关系”反馈关键词：出现这些才更可能值得改 KG
FACTY_PATTERNS = [
    r"缺少", r"补充", r"新增", r"遗漏", r"应包含", r"定义", r"属于", r"包括", r"组成",
    r"导致", r"影响", r"依赖", r"关系", r"关联", r"前提", r"因果", r"条件",
    r"missing", r"add", r"omit", r"should include", r"define", r"belongs to", r"include",
    r"consist", r"cause", r"affect", r"depend", r"relation", r"correlat", r"condition",
]


# =========================
# 工具函数
# =========================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def is_language_only_feedback(feedback: str) -> bool:
    fb = str(feedback)
    # 如果命中语言润色关键词且不命中事实关键词，就认为是 language-only
    lang_hit = any(re.search(p, fb, flags=re.IGNORECASE) for p in LANGUAGE_ONLY_PATTERNS)
    fact_hit = any(re.search(p, fb, flags=re.IGNORECASE) for p in FACTY_PATTERNS)
    return lang_hit and (not fact_hit)


def looks_like_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def build_entity_matcher(names):
    """
    返回一个列表 [(name_norm, regex_pattern), ...]
    - 长名字优先，减少短词误匹配
    - 英文用 \\b 边界，中文用“非字母数字”边界近似
    """
    # 去掉 stoplist & 空
    names = [n for n in names if n and (n not in STOPLIST)]

    # 长的优先
    names = sorted(set(names), key=len, reverse=True)

    compiled = []
    for n in names:
        if looks_like_chinese(n):
            # 中文：前后不是字母数字下划线（近似边界）
            pat = re.compile(rf"(?<![0-9a-zA-Z_]){re.escape(n)}(?![0-9a-zA-Z_])", flags=re.IGNORECASE)
        else:
            # 英文/数字：单词边界
            pat = re.compile(rf"\b{re.escape(n)}\b", flags=re.IGNORECASE)
        compiled.append((n, pat))
    return compiled


def extract_entities(text: str, matcher, name2id: dict):
    """
    从文本中找 KG 实体（更稳的 regex 边界匹配）
    返回 set(node_id)
    """
    found = set()
    text_norm = normalize(text)
    for name_norm, pat in matcher:
        if pat.search(text_norm):
            nid = name2id.get(name_norm)
            if nid:
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
    name2id = {normalize(row["name"]): row["node_id"] for _, row in nodes_df.iterrows()}
    id2name = {row["node_id"]: row["name"] for _, row in nodes_df.iterrows()}

    # 1.1 构建 matcher（长名字优先 + 边界匹配 + stoplist）
    matcher = build_entity_matcher(list(name2id.keys()))

    # 2. 读 Step10 改写后的题目
    q_df = pd.read_csv(REVISED_Q_TSV, sep="\t")

    required_cols = {"qid", "original_question", "revised_question", "changed", "revision_reason"}
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

        # 3.0 如果反馈只是语言润色，默认不建议改 KG（至少不加边）
        language_only = is_language_only_feedback(feedback)

        orig_entities = extract_entities(orig_q, matcher, name2id)
        rev_entities = extract_entities(rev_q, matcher, name2id)

        # 新出现的实体（rev 有 orig 无）
        added_entities = list(rev_entities - orig_entities)

        # 3.1 add_node：控制每题最多 MAX_NODES_PER_QID 个
        for nid in added_entities[:MAX_NODES_PER_QID]:
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

        # 3.2 add_edge：只连“新增实体”到“已有实体”，并限流
        # 目的：避免 rev_entities 全组合导致边爆炸
        if language_only:
            # 语言润色反馈：不建议加边（可保守些）
            continue

        existing_entities = list(rev_entities - set(added_entities))
        edge_count = 0

        for new_id in added_entities:
            if edge_count >= MAX_EDGES_PER_QID:
                break

            # 一个新增实体最多尝试连接到 1~2 个已有实体（可按需要调整）
            for old_id in existing_entities:
                if edge_count >= MAX_EDGES_PER_QID:
                    break

                if not has_edge(edges_df, new_id, old_id):
                    suggestions.append(
                        {
                            "qid": qid,
                            "action": "add_edge",
                            "entity1_id": new_id,
                            "entity1_name": id2name.get(new_id, ""),
                            "entity2_id": old_id,
                            "entity2_name": id2name.get(old_id, ""),
                            "relation_type_old": "",
                            "relation_type_new": "related_to",
                            "revision_reason": feedback,
                        }
                    )
                    edge_count += 1

    # 4. 输出
    out_df = pd.DataFrame(suggestions)
    out_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"[Step11] KG update suggestions saved to {OUTPUT_TSV}")
    print(f"[Step11] Total suggestions: {len(out_df)}")
    if len(out_df) > 0:
        print("[Step11] action counts:")
        print(out_df["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
