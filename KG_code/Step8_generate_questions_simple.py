"""
Step8 — 使用 LLM 从知识图谱 + 原始句子生成法律单选题（MCQ）

改动要点：
1. 新增读取 Step2 句子列表的 TSV（包含 sentence_id 和原始 text）
2. 在构造 facts 时，把 “KG 边 + 对应原句” 合在一起喂给 LLM
3. prompt 明确要求：题目必须能从【事实 + 原句】中推导
"""

import csv
import os
import json
import time
from typing import List, Dict

from openai import OpenAI

# ========== 路径配置 ==========

# Step4 输出：KG 节点、边
# NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_nodes.tsv"
# EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step4_output\第一讲_KG_edges.tsv"

NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_nodes_updated.tsv"
EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_edges_updated.tsv"

# ✅ 新增：Step2 输出的句子列表（需要包含 sentence_id / page_no / text）
SENT_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step2_output\第一讲_句子列表.tsv"

# 使用你之前写好的单选题 system prompt 文件
PROMPT_PATH = r"D:\Desktop\KG_allprocess\KG_code\prompt.txt"

# 输出：单选题 TSV
# OUTPUT_Q_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step8_output\第一讲_MCQ.tsv"

OUTPUT_Q_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step8_output\第一讲_MCQ_updated.tsv"


# ========== LLM 配置 ==========

# ✅ 建议使用环境变量存 key，避免明文写死在代码里
#   你可以：
#   - 在系统环境变量里设置：GITEE_AI_API_KEY=你的key
#   - 或者直接把 os.getenv(...) 替换成 "你的真实 key"，例如：
#       api_key="DUxxxxxxxxxxxxxxxxxxxx"
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ========== LLM 配置 ==========

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key = config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

# 模型名称：按你在 gitee 上真实可用的模型名称填写
MODEL_NAME = "DeepSeek-R1"  # TODO: 如有需要可以修改成其他模型


# ========== 生成策略参数 ==========

MAX_QUESTIONS = 50          # 最多生成多少题
EDGES_PER_CHUNK = 5         # 每次给 LLM 的 fact 条数
QUESTIONS_PER_CHUNK = 3     # 每个 chunk 期望生成几道题（上限值）


# ========== 工具函数 ==========

def load_prompt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT = load_prompt_text(PROMPT_PATH)


def load_nodes(path: str) -> Dict[str, Dict]:
    """读取节点 TSV：node_id -> row"""
    nodes: Dict[str, Dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            nodes[r["node_id"]] = r
    return nodes


def load_edges(path: str) -> List[Dict]:
    """读取边 TSV：返回列表"""
    edges: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            edges.append(r)
    return edges


def load_sentences(path: str) -> Dict[str, Dict]:
    """
    读取 Step2 的句子 TSV：
    假设列顺序为：sentence_id | page_no | text
    （和你 Step3 的读取方式保持一致，不依赖列名）
    """
    sentences: Dict[str, Dict] = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到句子 TSV 文件：{path}")

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过表头
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


def build_fact_strings(
    nodes: Dict[str, Dict],
    edges: List[Dict],
    sentences: Dict[str, Dict],
) -> List[str]:
    """
    把 KG 的边 + 对应原句一起变成“可读事实字符串”，喂给 LLM。

    格式示例：
      事实：全国人大常委会 --related_to--> 刑法修正案(八)
      来源原句：全国人大常委会通过了《刑法修正案(八)》。

    如果找不到 sentence_id 对应的原句，则只给事实行。
    """
    facts: List[str] = []
    for e in edges:
        src_name = nodes.get(e["src_id"], {}).get("name", e["src_id"])
        dst_name = nodes.get(e["dst_id"], {}).get("name", e["dst_id"])
        rel = e.get("relation_type", "related_to")

        sent_id = e.get("sentence_id", "")
        sent_text = sentences.get(sent_id, {}).get("text", "").strip()

        if sent_text:
            fact = (
                f"事实：{src_name} --{rel}--> {dst_name}\n"
                f"来源原句：{sent_text}"
            )
        else:
            fact = f"事实：{src_name} --{rel}--> {dst_name}"

        facts.append(fact)
    return facts


def chunk_list(lst: List[str], size: int) -> List[List[str]]:
    """
    把列表按固定大小切分成多个小块。
    返回 List[List[str]]。
    """
    chunks: List[List[str]] = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i:i + size])
    return chunks


# ========== LLM 调用：生成 MCQ ==========

def call_llm_for_mcq(fact_chunk: List[str], n_questions: int) -> List[Dict]:
    """
    调用 LLM：基于一个 fact_chunk 生成若干道单选题。

    返回格式：
    [
      {
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "answer": "B"
      },
      ...
    ]
    """
    if not fact_chunk or n_questions <= 0:
        return []

    # 加编号，方便模型阅读
    facts_text = "\n".join(
        f"{idx + 1}. {f}" for idx, f in enumerate(fact_chunk)
    )

    user_prompt = f"""
下面是若干条来自法律知识图谱的“事实及其来源原句”：

{facts_text}

说明：
- 每条记录通常包含两行：
  - “事实：...” 行描述知识图谱中的实体关系
  - “来源原句：...” 行给出该事实在原始材料中的完整句子（若有）

请你【仅根据上述事实及原句】生成 {n_questions} 道中文法律单选题（Multiple Choice Questions，MCQ），并满足以下要求：

1. 每道题必须包含字段：
   - "question"：题干（用中文表述，可适当改写原句，但不能脱离原意）
   - "options"：包含四个元素的数组 ["A. ...", "B. ...", "C. ...", "D. ..."]
   - "answer"：正确选项的选项字母（只能是 "A"、"B"、"C" 或 "D"）
2. 题目内容和选项必须能够从【事实 + 原句】中推导出来，不允许引入材料中没有的信息。
3. 四个选项都要合理、有一定迷惑性，但不能明显错误或与原句矛盾。
4. 每道题只能有一个唯一正确答案，不要出现多选或模糊不清的情况。
5. 不要输出题目解析或任何解释。
6. 最终输出必须是一个 JSON 数组，例如：
   [
     {{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "B"}},
     ...
   ]
7. 不要在 JSON 外输出任何额外文字（不要 markdown、不要说明，只要 JSON）。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    # 解析 JSON（兼容模型可能加的 ```json ... ``` 包裹）
    try:
        if content.startswith("```"):
            # 去掉可能的 ```json / ``` 包裹
            content = content.strip("`")
            idx = content.find("[")
            if idx != -1:
                content = content[idx:]

        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            content = content[start: end + 1]

        data = json.loads(content)
        mcqs: List[Dict] = []

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                options = item.get("options", [])
                answer = str(item.get("answer", "")).strip()

                if not q or not isinstance(options, list) or len(options) != 4:
                    continue

                options = [str(opt).strip() for opt in options]
                if answer not in ("A", "B", "C", "D"):
                    continue

                mcqs.append(
                    {
                        "question": q,
                        "options": options,
                        "answer": answer,
                    }
                )

        return mcqs[:n_questions]

    except Exception as e:
        print("⚠ 解析 LLM 输出 JSON 失败：", e)
        print("原始内容片段：", content[:300], "...")
        return []


# ========== 主逻辑：分块生成 MCQ + 进度显示 ==========

def generate_mcq_with_llm(
    nodes: Dict[str, Dict],
    edges: List[Dict],
    sentences: Dict[str, Dict],
) -> List[Dict]:
    """
    分块喂 facts（KG 边 + 原句），调用 LLM 生成多道单选题。

    返回格式：
    [
      {
        "qid": "q0001",
        "question": "...",
        "option_a": "...",
        "option_b": "...",
        "option_c": "...",
        "option_d": "...",
        "answer": "B"
      },
      ...
    ]
    """
    facts = build_fact_strings(nodes, edges, sentences)
    chunks = chunk_list(facts, EDGES_PER_CHUNK)

    all_mcq_rows: List[Dict] = []
    q_counter = 1
    total_chunks = len(chunks)
    avg_call_time: List[float] = []

    print(f"\n🔄 共 {total_chunks} 个 fact chunk，将生成最多 {MAX_QUESTIONS} 道题\n")

    for idx, fact_chunk in enumerate(chunks, start=1):
        if len(all_mcq_rows) >= MAX_QUESTIONS:
            break

        remain = MAX_QUESTIONS - len(all_mcq_rows)
        n_q = min(QUESTIONS_PER_CHUNK, remain)

        print(f"\n📌 Chunk {idx}/{total_chunks}: 尝试生成 {n_q} 道题")
        print("   🤖 调用 LLM 中……")

        start_time = time.time()
        mcqs = call_llm_for_mcq(fact_chunk, n_q)
        cost = time.time() - start_time
        avg_call_time.append(cost)

        print(f"   ✅ LLM 返回（耗时 {cost:.2f} 秒）")

        for item in mcqs:
            qid = f"q{q_counter:04d}"
            options = item["options"]
            row = {
                "qid": qid,
                "question": item["question"],
                "option_a": options[0],
                "option_b": options[1],
                "option_c": options[2],
                "option_d": options[3],
                "answer": item["answer"],
            }
            all_mcq_rows.append(row)
            q_counter += 1

        # 简单 ETA 估计
        progress = len(all_mcq_rows)
        if avg_call_time and progress > 0:
            avg_t = sum(avg_call_time) / len(avg_call_time)
            # 剩余要调用几次 LLM ≈ 剩余题数 / 每次生成题数
            remain_calls = max(
                (MAX_QUESTIONS - progress) / max(QUESTIONS_PER_CHUNK, 1),
                0,
            )
            eta = remain_calls * avg_t
            print(
                f"   📊 进度：已生成 {progress}/{MAX_QUESTIONS} 道题 | "
                f"预估剩余时间约：{eta:.1f} 秒"
            )

    print("\n🎉 题目生成完成！")
    return all_mcq_rows


# ========== 保存 TSV ==========

def save_mcq(rows: List[Dict], path: str):
    """保存 TSV：qid, question, option_a, option_b, option_c, option_d, answer"""
    if not rows:
        print("⚠ 没有生成任何题目，文件不会写出。")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "qid",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "answer",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ 已保存单选题：{path}（共 {len(rows)} 题）")


# ========== main ==========

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
