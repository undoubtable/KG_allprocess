"""
Step8 — 使用 LLM 从知识图谱 + 原始句子生成法律单选题（MCQ）

✅ v2 改动要点（关键）：
1) 生成时要求 LLM 为每题返回 fact_index（使用了第几条事实）
2) 写出 TSV 时新增两列：
   - kg_fact:  src_id|relation_type|dst_id   （可用于严格审计 C/D）
   - context:  对应 sentence_id 的原句（可用于 R1 judge 与追溯）
3) 仍保留 chunk 调用（效率更高），但每题可追溯到具体 KG edge

输出字段：
qid, question, option_a, option_b, option_c, option_d, answer, kg_fact, context
"""

import csv
import os
import json
import time
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from pipeline_config import STEP4_NODES_TSV, STEP4_EDGES_TSV, STEP2_SENT_TSV
from pipeline_config import STEP8_Q_TSV, PROMPT_PATH

import yaml


# ========== 路径配置 ==========
NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)
SENT_TSV = str(STEP2_SENT_TSV)
OUTPUT_Q_TSV = str(STEP8_Q_TSV)

# ========== LLM 配置 ==========
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

# ✅ 题目生成推荐用 DeepSeek-V3（或 v3.2 思考关）
MODEL_NAME = config.get("qg_model", "DeepSeek-V3")

# ========== 生成策略参数 ==========
MAX_QUESTIONS = 50
EDGES_PER_CHUNK = 5
QUESTIONS_PER_CHUNK = 3

# ========== 工具函数 ==========
def load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_PROMPT = load_prompt_text(PROMPT_PATH)

def load_nodes(path: str) -> Dict[str, Dict[str, str]]:
    nodes: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            nodes[r["node_id"]] = r
    return nodes

def load_edges(path: str) -> List[Dict[str, str]]:
    edges: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            edges.append(r)
    return edges

def load_sentences(path: str) -> Dict[str, Dict[str, str]]:
    """
    读取 Step2 的句子 TSV：
    假设列顺序为：sentence_id | page_no | text
    """
    sentences: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到句子 TSV 文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()  # 跳过表头
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sentence_id = parts[0]
            page_no = parts[1]
            text = parts[2]
            sentences[sentence_id] = {
                "sentence_id": sentence_id,
                "page_no": page_no,
                "text": text,
            }
    return sentences

def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    chunks: List[List[Any]] = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])
    return chunks

def build_fact_items(
    nodes: Dict[str, Dict[str, str]],
    edges: List[Dict[str, str]],
    sentences: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    每条 edge 构造成一个 fact_item：
    - display_text: 给 LLM 看的人类可读事实 + 原句
    - kg_fact: src_id|rel|dst_id（用于严格评估）
    - context: 原句
    """
    items: List[Dict[str, str]] = []
    for e in edges:
        src_id = e["src_id"]
        dst_id = e["dst_id"]
        rel = e.get("relation_type", "related_to")

        src_name = nodes.get(src_id, {}).get("name", src_id)
        dst_name = nodes.get(dst_id, {}).get("name", dst_id)

        sent_id = e.get("sentence_id", "")
        sent_text = sentences.get(sent_id, {}).get("text", "").strip()

        display = f"事实：{src_name} --{rel}--> {dst_name}"
        if sent_text:
            display += f"\n来源原句：{sent_text}"

        items.append({
            "display_text": display,
            "kg_fact": f"{src_id}|{rel}|{dst_id}",
            "context": sent_text,
        })
    return items


# ========== LLM 调用：生成 MCQ（每题返回 fact_index） ==========
def call_llm_for_mcq(fact_items: List[Dict[str, str]], n_questions: int) -> List[Dict[str, Any]]:
    """
    返回格式（强制）：
    [
      {
        "fact_index": 1,
        "question": "...",
        "options": ["A. ...","B. ...","C. ...","D. ..."],
        "answer": "B"
      },
      ...
    ]
    其中 fact_index 指向本 chunk 中编号的事实（从 1 开始）
    """
    if not fact_items or n_questions <= 0:
        return []

    facts_text = "\n".join(
        f"{idx+1}. {it['display_text']}" for idx, it in enumerate(fact_items)
    )

    user_prompt = f"""
下面是若干条来自法律知识图谱的“事实及其来源原句”（已编号）：

{facts_text}

请你【仅根据上述事实及原句】生成 {n_questions} 道中文法律单选题（MCQ），并满足：

1) 每道题必须包含字段：
   - "fact_index"：整数，表示本题使用了上面第几条事实（从1开始）
   - "question"：题干
   - "options"：四个元素的数组 ["A. ...","B. ...","C. ...","D. ..."]
   - "answer"：正确选项字母（A/B/C/D）

2) 题干与正确答案必须能从对应的那条事实推导出来；不得引入材料外信息。
3) 选项需要迷惑性，但不能明显错误或与原句矛盾。
4) 每题只有一个正确答案。
5) 只输出 JSON 数组，不要输出 markdown 或解释。

输出示例：
[
  {{"fact_index": 2, "question": "...", "options": ["A. ...","B. ...","C. ...","D. ..."], "answer": "B"}}
]
""".strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        # 兼容 ```json 包裹
        if content.startswith("```"):
            content = content.strip("`")
            lb = content.find("[")
            if lb != -1:
                content = content[lb:]
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            content = content[start:end+1]

        data = json.loads(content)
        out: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                fi = item.get("fact_index", None)
                q = str(item.get("question", "")).strip()
                options = item.get("options", [])
                ans = str(item.get("answer", "")).strip().upper()

                if not isinstance(fi, int) or fi < 1 or fi > len(fact_items):
                    continue
                if not q or not isinstance(options, list) or len(options) != 4:
                    continue
                options = [str(x).strip() for x in options]
                if ans not in ("A", "B", "C", "D"):
                    continue

                out.append({
                    "fact_index": fi,
                    "question": q,
                    "options": options,
                    "answer": ans,
                })
        return out[:n_questions]
    except Exception as e:
        print("⚠ 解析 LLM 输出 JSON 失败：", e)
        print("原始内容片段：", content[:300], "...")
        return []


# ========== 主逻辑：分块生成 + 写出可审计字段 ==========
def generate_mcq_with_llm(
    nodes: Dict[str, Dict[str, str]],
    edges: List[Dict[str, str]],
    sentences: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    fact_items = build_fact_items(nodes, edges, sentences)
    chunks = chunk_list(fact_items, EDGES_PER_CHUNK)

    all_rows: List[Dict[str, str]] = []
    q_counter = 1

    print(f"\n🔄 共 {len(chunks)} 个 chunk，将生成最多 {MAX_QUESTIONS} 道题\n")

    avg_call_time: List[float] = []
    for chunk_idx, fact_chunk in enumerate(chunks, start=1):
        if len(all_rows) >= MAX_QUESTIONS:
            break

        remain = MAX_QUESTIONS - len(all_rows)
        n_q = min(QUESTIONS_PER_CHUNK, remain)

        print(f"\n📌 Chunk {chunk_idx}/{len(chunks)}：尝试生成 {n_q} 道题")
        start_time = time.time()
        mcqs = call_llm_for_mcq(fact_chunk, n_q)
        cost = time.time() - start_time
        avg_call_time.append(cost)
        print(f"   ✅ 返回 {len(mcqs)} 道（耗时 {cost:.2f}s）")

        for item in mcqs:
            qid = f"q{q_counter:04d}"
            opts = item["options"]
            fi = item["fact_index"] - 1  # 0-based
            kg_fact = fact_chunk[fi]["kg_fact"]
            context = fact_chunk[fi]["context"]

            row = {
                "qid": qid,
                "question": item["question"],
                "option_a": opts[0],
                "option_b": opts[1],
                "option_c": opts[2],
                "option_d": opts[3],
                "answer": item["answer"],
                "kg_fact": kg_fact,
                "context": context,
            }
            all_rows.append(row)
            q_counter += 1

        # ETA
        if avg_call_time and len(all_rows) > 0:
            avg_t = sum(avg_call_time) / len(avg_call_time)
            remain_calls = max((MAX_QUESTIONS - len(all_rows)) / max(QUESTIONS_PER_CHUNK, 1), 0)
            eta = remain_calls * avg_t
            print(f"   📊 进度：{len(all_rows)}/{MAX_QUESTIONS} | ETA≈{eta:.1f}s")

    print("\n🎉 题目生成完成！")
    return all_rows


def save_mcq(rows: List[Dict[str, str]], path: str):
    if not rows:
        print("⚠ 没有生成任何题目，文件不会写出。")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "qid", "question",
        "option_a", "option_b", "option_c", "option_d",
        "answer",
        "kg_fact",     # ✅ 新增
        "context",     # ✅ 新增
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ 已保存单选题：{path}（共 {len(rows)} 题）")


def main():
    if not os.path.exists(NODES_TSV) or not os.path.exists(EDGES_TSV):
        raise FileNotFoundError("请检查节点/边 TSV 路径是否正确")
    if not os.path.exists(SENT_TSV):
        raise FileNotFoundError("请检查句子 TSV 路径是否正确")

    nodes = load_nodes(NODES_TSV)
    edges = load_edges(EDGES_TSV)
    sentences = load_sentences(SENT_TSV)

    print(f"📄 已加载节点数：{len(nodes)}，边数：{len(edges)}，句子数：{len(sentences)}")
    mcq_rows = generate_mcq_with_llm(nodes, edges, sentences)
    save_mcq(mcq_rows, OUTPUT_Q_TSV)


if __name__ == "__main__":
    main()
