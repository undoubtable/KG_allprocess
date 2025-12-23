import os
import csv
import json
import yaml
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI

from pipeline_config import STEP2_SENT_TSV, STEP3_ENT_TSV, STEP4_NODES_TSV, STEP4_EDGES_TSV

sent_tsv_path = str(STEP2_SENT_TSV)
entity_tsv_path = str(STEP3_ENT_TSV)
out_nodes_path = str(STEP4_NODES_TSV)
out_edges_path = str(STEP4_EDGES_TSV)

# ======== LLM 配置（沿用你的 Gitee AI OpenAI兼容调用）========
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

MODEL_NAME = "DeepSeek-V3"

# ======== 关系类型集合（KG-version1 先用小集合，便于控噪）========
REL_TYPES = [
    "defines",      # 定义/解释
    "includes",     # 包含/组成
    "part_of",      # 属于/隶属
    "causes",       # 导致/引起
    "applies_to",   # 适用/针对
    "related_to"    # 兜底（不确定时）
]

# 控制 LLM 调用量：每句只对相邻实体对做一次关系判别
MAX_EDGES_PER_SENT = 20  # 句子太长时，最多处理前 N 对相邻实体

DEFAULT_EDGE_CONF = 0.80


def load_sentences(tsv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取 Step2 句子 TSV：sentence_id | page_no | text
    """
    sents = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, page_no, text = parts[0], int(parts[1]), parts[2]
            sents[sid] = {"sentence_id": sid, "page_no": page_no, "text": text}
    return sents


def load_entities(tsv_path: str) -> List[Dict[str, Any]]:
    """
    读取 Step3 的实体 TSV
    entity_id, sentence_id, page_no, mention, start_char, end_char, ent_type, confidence
    """
    entities = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["page_no"] = int(row.get("page_no", 0))
            row["start_char"] = int(row.get("start_char", 0))
            row["end_char"] = int(row.get("end_char", 0))
            try:
                row["confidence"] = float(row.get("confidence", 0.0))
            except ValueError:
                row["confidence"] = 0.0
            entities.append(row)
    return entities


def build_unique_nodes(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
    """
    按 (mention, ent_type) 合并实体，生成唯一节点（与你原 Step4 一致）:contentReference[oaicite:2]{index=2}
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


def _extract_json(content: str) -> Optional[Dict[str, Any]]:
    content = (content or "").strip()
    l = content.find("{")
    r = content.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(content[l:r + 1])
    except Exception:
        return None


def llm_classify_relation(sentence: str, head: Dict[str, Any], tail: Dict[str, Any]) -> Tuple[str, float]:
    """
    给定句子与两个实体（相邻），让 LLM 选择 relation_type
    输出：relation_type, confidence
    """
    h_m = head["mention"]
    t_m = tail["mention"]
    h_t = head.get("ent_type", "Other")
    t_t = tail.get("ent_type", "Other")

    system_prompt = (
        "你是中文关系抽取助手。给定句子和两个实体，请判断二者在句子中是否存在明确关系。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"relation_type\":\"...\",\"confidence\":0.0}\n"
        "规则：\n"
        f"1) relation_type 只能从 {REL_TYPES} 中选择。\n"
        "2) 如果关系不明确，请输出 related_to，并给较低 confidence。\n"
        "3) confidence 取 0~1，越确定越高。\n"
        "4) 不要编造句子外知识，只根据句子。\n"
    )

    user_prompt = (
        f"句子：{sentence}\n"
        f"实体A：{h_m}（{h_t}）\n"
        f"实体B：{t_m}（{t_t}）\n"
        "请输出 JSON："
    )

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.1,
    )

    data = _extract_json(resp.choices[0].message.content)
    if not data:
        return "related_to", DEFAULT_EDGE_CONF

    rel = str(data.get("relation_type", "related_to")).strip()
    if rel not in REL_TYPES:
        rel = "related_to"

    try:
        conf = float(data.get("confidence", DEFAULT_EDGE_CONF))
    except Exception:
        conf = DEFAULT_EDGE_CONF

    # clamp
    conf = max(0.0, min(1.0, conf))
    return rel, conf


def build_edges_by_sentence_llm(
    sentences: Dict[str, Dict[str, Any]],
    entities: List[Dict[str, Any]],
    key2node_id: Dict[Tuple[str, str], str],
) -> List[Dict[str, Any]]:
    """
    KG-version1：同句内按 start_char 排序，只对相邻实体对连边（结构与原 Step4 一致）:contentReference[oaicite:3]{index=3}
    但 relation_type 由 LLM 判别。
    """
    ents_by_sent = defaultdict(list)
    for e in entities:
        ents_by_sent[e["sentence_id"]].append(e)

    edges = []
    edge_idx = 1

    for si, (sent_id, ents) in enumerate(ents_by_sent.items(), start=1):
        sent = sentences.get(sent_id)
        if not sent:
            continue

        ents_sorted = sorted(ents, key=lambda x: x["start_char"])
        if len(ents_sorted) < 2:
            continue

        page_no = sent["page_no"]
        text = sent["text"]

        pair_count = 0
        for i in range(len(ents_sorted) - 1):
            if pair_count >= MAX_EDGES_PER_SENT:
                break

            h = ents_sorted[i]
            t = ents_sorted[i + 1]

            h_key = (h["mention"], h.get("ent_type", "Entity"))
            t_key = (t["mention"], t.get("ent_type", "Entity"))

            src = key2node_id.get(h_key)
            dst = key2node_id.get(t_key)
            if not src or not dst or src == dst:
                continue

            rel_type, rel_conf = llm_classify_relation(text, h, t)

            edges.append(
                {
                    "edge_id": f"e{edge_idx:05d}",
                    "src_id": src,
                    "dst_id": dst,
                    "relation_type": rel_type,
                    "page_no": page_no,
                    "sentence_id": sent_id,
                    # 边置信度 = min(实体置信度) 与 关系置信度 的组合（保守）
                    "confidence": min(h["confidence"], t["confidence"], rel_conf),
                }
            )
            edge_idx += 1
            pair_count += 1

        if si % 50 == 0:
            print(f"…已处理句子 {si}/{len(ents_by_sent)}，当前边数：{len(edges)}")

    return edges


def save_tsv(rows: List[Dict[str, Any]], fieldnames: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"✅ 已保存：{path}（{len(rows)} 行）")


def main():
    print("========== Step4 KG-version1（LLM 关系抽取）==========")

    # 1) 读取句子（用于 LLM 做关系判别）
    sentences = load_sentences(sent_tsv_path)
    print(f"📄 句子总数：{len(sentences)}")

    # 2) 读取实体
    entities = load_entities(entity_tsv_path)
    print(f"📄 实体总数（包括重复提及）：{len(entities)}")

    # 3) 合并成唯一节点
    nodes, key2node_id = build_unique_nodes(entities)
    print(f"✨ 合并后唯一实体（节点）数量：{len(nodes)}")

    # 4) 句子内相邻实体 → LLM 判别关系类型 → 生成边
    print(f"🤖 使用模型：{MODEL_NAME} 进行 relation_type 判别")
    edges = build_edges_by_sentence_llm(sentences, entities, key2node_id)
    print(f"🔗 生成边数量：{len(edges)}")

    # 5) 打印边样例
    print("\n📌 边示例（前10条）：")
    for e in edges[:10]:
        print(e)

    # 6) 保存
    save_tsv(nodes, ["node_id", "name", "label", "page_no", "sentence_id"], out_nodes_path)
    save_tsv(edges, ["edge_id", "src_id", "dst_id", "relation_type", "page_no", "sentence_id", "confidence"], out_edges_path)

    print("========== Done ==========")


if __name__ == "__main__":
    main()
