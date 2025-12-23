import os
import csv
import json
import yaml
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from openai import OpenAI

from pipeline_config import STEP2_SENT_TSV, STEP35_TRUTH_ENT_TSV, STEP45_NODES_TSV, STEP45_EDGES_TSV

# ===================== 路径配置 =====================
sent_tsv_path = str(STEP2_SENT_TSV)

# Truth-实体输出（来自 Truth_Entity_verbose / Truth_Entity_fast）
truth_entity_tsv_path = str(STEP35_TRUTH_ENT_TSV)

# Truth-关系输出（nodes / edges）
nodes_truth_path = str(STEP45_NODES_TSV)
edges_truth_path = str(STEP45_EDGES_TSV)
# ===================== LLM 配置（沿用你的 Gitee AI OpenAI兼容调用） =====================
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

CANDIDATE_MODEL = "DeepSeek-V3"
VERIFY_MODEL = "DeepSeek-R1"

# ===================== 关系类型集合（truth 推荐小而稳） =====================
REL_TYPES = [
    "defines",      # 定义/解释/属于…的含义
    "includes",     # 包含/组成/包括
    "part_of",      # 属于/隶属/构成…的一部分
    "causes",       # 导致/引起
    "applies_to",   # 适用/针对
    "punishes",     # 处罚/定罪（刑法/罪名场景）
    "related_to",   # 兜底（truth 一般会尽量不保留）
]

# ===================== FAST 核心参数 =====================
BATCH_SIZE = 8  # ✅ 每批句子数（推荐 6~12，根据接口稳定性调整）
PRINT_FIRST_N_BATCH = 1
PRINT_EVERY_N_BATCH = 5

# 句子内控量（很关键，避免 prompt 爆长/变慢）
MAX_MENTIONS_PER_SENT = 10       # 每句最多用多少个实体参与关系抽取
MAX_REL_CAND_PER_SENT = 10       # 每句 V3 最多候选关系
MAX_REL_KEEP_PER_SENT = 6        # 每句 R1 最多保留关系

EDGE_CONF_TRUTH = 0.95

# 限制“候选关系总量”，避免 batch payload 过大
MAX_TOTAL_CAND_REL_PER_BATCH = 120

# 重试配置
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 1.2

# ===================== I/O 读取 =====================
def load_sentences(tsv_path: str) -> Dict[str, Dict[str, Any]]:
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
    ents = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["page_no"] = int(row.get("page_no", 0))
            row["start_char"] = int(row.get("start_char", 0))
            row["end_char"] = int(row.get("end_char", 0))
            try:
                row["confidence"] = float(row.get("confidence", 0.0) or 0.0)
            except Exception:
                row["confidence"] = 0.0
            ents.append(row)
    return ents


# ===================== JSON 提取 =====================
def extract_json(content: str) -> Optional[Dict[str, Any]]:
    content = (content or "").strip()
    l = content.find("{")
    r = content.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(content[l:r + 1])
    except Exception:
        return None


# ===================== 可靠请求封装（带重试） =====================
def chat_with_retry(model: str, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            sleep_s = RETRY_BASE_SLEEP * attempt
            print(f"⚠️ LLM 请求失败（{model}）attempt {attempt}/{MAX_RETRIES}: {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM 请求失败（重试仍失败）：{last_err}")


# ===================== Truth 节点合并（与 Step4 一致） =====================
def build_unique_nodes(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
    key2node_id = {}
    nodes = []
    idx = 1

    for e in entities:
        key = (e["mention"], e.get("ent_type", "Other"))
        if key not in key2node_id:
            node_id = f"n{idx:05d}"
            key2node_id[key] = node_id
            idx += 1
            nodes.append({
                "node_id": node_id,
                "name": e["mention"],
                "label": e.get("ent_type", "Other"),
                "page_no": e.get("page_no", ""),
                "sentence_id": e.get("sentence_id", ""),
            })

    return nodes, key2node_id


# ===================== Batch：V3 候选关系 =====================
def candidate_relations_batch(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """
    items: [{"sentence_id","text","mentions"}...]
    return: {sentence_id: [{"head","rel","tail"}, ...], ...}
    """
    system = (
        "你是中文关系抽取助手。请对多个句子分别抽取实体关系。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"items\":[{\"sentence_id\":\"...\",\"relations\":[{\"head\":\"...\",\"rel\":\"...\",\"tail\":\"...\"}, ...]}, ...]}\n"
        "硬规则：\n"
        "1) head/tail 必须严格从该句的实体列表中选择（完全一致）。\n"
        f"2) rel 只能从 {REL_TYPES} 中选择。\n"
        "3) 只抽取句子里有明确语言证据支持的关系；不确定不要输出。\n"
        f"4) 每句最多输出 {MAX_REL_CAND_PER_SENT} 条关系。\n"
    )
    payload = {
        "items": [
            {"sentence_id": x["sentence_id"], "text": x["text"], "mentions": x["mentions"]}
            for x in items
        ]
    }

    content = chat_with_retry(
        model=CANDIDATE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
    )

    data = extract_json(content) or {}
    out: Dict[str, List[Dict[str, str]]] = {}
    items_out = data.get("items", [])
    if not isinstance(items_out, list):
        return out

    for it in items_out:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sentence_id", "")).strip()
        rels = it.get("relations", [])
        if not sid or not isinstance(rels, list):
            continue
        cleaned = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            h = str(r.get("head", "")).strip()
            rel = str(r.get("rel", "")).strip()
            t = str(r.get("tail", "")).strip()
            if h and t and rel:
                cleaned.append({"head": h, "rel": rel, "tail": t})
        out[sid] = cleaned[:MAX_REL_CAND_PER_SENT]
    return out


# ===================== Batch：R1 校验关系 =====================
def verify_relations_batch(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """
    items: [{"sentence_id","text","mentions","candidates":[{head,rel,tail}...]}...]
    return: {sentence_id: [{"head","rel","tail"}, ...], ...}
    """
    system = (
        "你是关系校验器。请对多个句子分别从候选关系中筛选应保留的关系。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"items\":[{\"sentence_id\":\"...\",\"relations\":[{\"head\":\"...\",\"rel\":\"...\",\"tail\":\"...\"}, ...]}, ...]}\n"
        "硬规则：\n"
        "1) 只能从该句的候选关系中选择（head/rel/tail 必须完全一致）。\n"
        f"2) rel 只能从 {REL_TYPES} 中选择。\n"
        "3) 只保留句子里有明确证据支持的关系；证据不足不要保留。\n"
        f"4) 每句最多保留 {MAX_REL_KEEP_PER_SENT} 条关系。\n"
        "宁可少留，不要错留。\n"
    )
    payload = {"items": items}

    content = chat_with_retry(
        model=VERIFY_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
    )

    data = extract_json(content) or {}
    out: Dict[str, List[Dict[str, str]]] = {}
    items_out = data.get("items", [])
    if not isinstance(items_out, list):
        return out

    for it in items_out:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sentence_id", "")).strip()
        rels = it.get("relations", [])
        if not sid or not isinstance(rels, list):
            continue
        cleaned = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            h = str(r.get("head", "")).strip()
            rel = str(r.get("rel", "")).strip()
            t = str(r.get("tail", "")).strip()
            if h and t and rel:
                cleaned.append({"head": h, "rel": rel, "tail": t})
        out[sid] = cleaned[:MAX_REL_KEEP_PER_SENT]
    return out


# ===================== 保存 TSV =====================
def save_tsv(rows: List[Dict[str, Any]], fields: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ saved: {path} ({len(rows)} rows)")


# ===================== 主流程 =====================
def main():
    print("========== Truth-Relation FAST（batch：V3 候选 + R1 校验）==========")
    print(f"🤖 候选模型：{CANDIDATE_MODEL} | 校验模型：{VERIFY_MODEL}")
    print(f"⚙️ BATCH_SIZE={BATCH_SIZE}")
    print(f"⚙️ MAX_MENTIONS_PER_SENT={MAX_MENTIONS_PER_SENT}, MAX_REL_CAND_PER_SENT={MAX_REL_CAND_PER_SENT}, MAX_REL_KEEP_PER_SENT={MAX_REL_KEEP_PER_SENT}")
    print(f"📌 REL_TYPES={REL_TYPES}\n")

    sentences = load_sentences(sent_tsv_path)
    print(f"📄 已加载句子数：{len(sentences)}")

    entities = load_entities(truth_entity_tsv_path)
    print(f"📄 已加载 truth 实体数：{len(entities)}")

    # nodes（按 mention+type 合并）
    nodes, key2node_id = build_unique_nodes(entities)
    print(f"✨ 合并后 truth 节点数：{len(nodes)}")

    # sentence -> entities（按 start_char 排序）
    ents_by_sent = defaultdict(list)
    for e in entities:
        ents_by_sent[e["sentence_id"]].append(e)

    # 统一成可迭代列表（只处理有 >=2 个实体的句子）
    sent_items = []
    for sid, ents in ents_by_sent.items():
        sent = sentences.get(sid)
        if not sent:
            continue
        if len(ents) < 2:
            continue
        ents_sorted = sorted(ents, key=lambda x: x["start_char"])

        # mentions 去重保序 + 截断
        mentions = []
        mention2type = {}
        for e in ents_sorted:
            m = e["mention"]
            if m not in mentions:
                mentions.append(m)
            mention2type.setdefault(m, e.get("ent_type", "Other"))
        if len(mentions) > MAX_MENTIONS_PER_SENT:
            mentions = mentions[:MAX_MENTIONS_PER_SENT]

        sent_items.append({
            "sentence_id": sid,
            "page_no": sent["page_no"],
            "text": sent["text"],
            "mentions": mentions,
            "mention2type": mention2type,  # 之后映射 node_id 用
        })

    print(f"🧩 满足关系抽取条件的句子数（>=2 实体）：{len(sent_items)}")
    total_batches = (len(sent_items) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"🚀 将分 {total_batches} 个 batch 处理（每 batch {BATCH_SIZE} 句）\n")

    edges: List[Dict[str, Any]] = []
    seen_edge = set()
    edge_idx = 1

    # batch 循环
    for bi in range(total_batches):
        batch = sent_items[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]

        # -------- 1) V3 候选关系（batch 一次）--------
        cand_input = [{"sentence_id": x["sentence_id"], "text": x["text"], "mentions": x["mentions"]} for x in batch]
        cand_map = candidate_relations_batch(cand_input)

        # -------- 2) 清洗候选关系 + 控制 batch 总量 --------
        verify_items = []
        total_cand_rel = 0

        for x in batch:
            sid = x["sentence_id"]
            mentions = x["mentions"]
            mention_set = set(mentions)

            cand = cand_map.get(sid, []) or []
            # 强约束：head/tail 必须在 mentions 内，rel 合法，且 head!=tail
            cand2 = []
            for r in cand:
                h, rel, t = r["head"], r["rel"], r["tail"]
                if h in mention_set and t in mention_set and rel in REL_TYPES and h != t:
                    cand2.append({"head": h, "rel": rel, "tail": t})

            # 去重保序
            seen_local = set()
            dedup = []
            for r in cand2:
                k = (r["head"], r["rel"], r["tail"])
                if k not in seen_local:
                    seen_local.add(k)
                    dedup.append(r)

            # 每句截断
            dedup = dedup[:MAX_REL_CAND_PER_SENT]

            # batch 总量控制：太多就再砍
            if total_cand_rel + len(dedup) > MAX_TOTAL_CAND_REL_PER_BATCH:
                remain = max(0, MAX_TOTAL_CAND_REL_PER_BATCH - total_cand_rel)
                dedup = dedup[:remain]

            total_cand_rel += len(dedup)

            verify_items.append({
                "sentence_id": sid,
                "text": x["text"],
                "mentions": mentions,
                # 注意：校验时让模型只从候选里选
                "candidates": dedup,
            })

        # -------- 3) R1 校验关系（batch 一次）--------
        keep_map = verify_relations_batch(verify_items)

        # -------- 4) 生成 edges --------
        # 打印样例（前几个 batch）
        if bi < PRINT_FIRST_N_BATCH:
            print(f"=== batch {bi+1}/{total_batches} 示例 ===")
            for x in batch[:3]:
                sid = x["sentence_id"]
                cand_show = cand_map.get(sid, [])
                keep_show = keep_map.get(sid, [])
                print(f"[sid={sid}]")
                print("TEXT:", x["text"])
                print("MENTIONS:", x["mentions"])
                print("CAND_REL:", [f"{r['head']}--{r['rel']}-->{r['tail']}" for r in cand_show[:8]])
                print("KEEP_REL:", [f"{r['head']}--{r['rel']}-->{r['tail']}" for r in keep_show[:8]])
                print()

        for x in batch:
            sid = x["sentence_id"]
            page_no = x["page_no"]
            mention2type = x["mention2type"]
            mentions = x["mentions"]

            # mention -> node_id
            mention2nid = {}
            for m in mentions:
                t = mention2type.get(m, "Other")
                nid = key2node_id.get((m, t))
                if nid:
                    mention2nid[m] = nid

            # 候选 set（用于强约束：keep 必须来自 candidates）
            cand_list = (verify_items[[i for i, it in enumerate(verify_items) if it["sentence_id"] == sid][0]]["candidates"]
                         if any(it["sentence_id"] == sid for it in verify_items) else [])
            cand_set = {(r["head"], r["rel"], r["tail"]) for r in cand_list}

            keep = keep_map.get(sid, []) or []
            # 强约束：只保留来自候选的
            keep2 = []
            for r in keep:
                k = (r.get("head"), r.get("rel"), r.get("tail"))
                if k in cand_set and r.get("rel") in REL_TYPES:
                    keep2.append({"head": r["head"], "rel": r["rel"], "tail": r["tail"]})

            # 去重保序 + 每句截断
            seen_local = set()
            final_keep = []
            for r in keep2:
                k = (r["head"], r["rel"], r["tail"])
                if k not in seen_local:
                    seen_local.add(k)
                    final_keep.append(r)
            final_keep = final_keep[:MAX_REL_KEEP_PER_SENT]

            for r in final_keep:
                src = mention2nid.get(r["head"])
                dst = mention2nid.get(r["tail"])
                if not src or not dst or src == dst:
                    continue
                k = (src, r["rel"], dst, sid)
                if k in seen_edge:
                    continue
                seen_edge.add(k)

                edges.append({
                    "edge_id": f"e{edge_idx:05d}",
                    "src_id": src,
                    "dst_id": dst,
                    "relation_type": r["rel"],
                    "page_no": page_no,
                    "sentence_id": sid,
                    "confidence": EDGE_CONF_TRUTH,
                })
                edge_idx += 1

        if (bi + 1) % PRINT_EVERY_N_BATCH == 0:
            print(f"…已处理 batch {bi+1}/{total_batches} | truth_edges={len(edges)}")

    print("\n📌 truth 边示例（前10条）：")
    for e in edges[:10]:
        print(e)

    # 保存
    save_tsv(nodes, ["node_id", "name", "label", "page_no", "sentence_id"], nodes_truth_path)
    save_tsv(edges, ["edge_id", "src_id", "dst_id", "relation_type", "page_no", "sentence_id", "confidence"], edges_truth_path)

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()
