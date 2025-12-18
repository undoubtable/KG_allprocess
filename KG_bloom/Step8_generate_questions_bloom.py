"""
Step8 — 使用 LLM 从知识图谱 + 原始句子生成法律单选题（MCQ）
新增：基于布鲁姆分类（Bloom's Taxonomy）生成题目，并在输出中标注 bloom_level / bloom_label

要点：
1) facts = KG 边 + 对应原句
2) LLM 输出新增 bloom_level/bloom_label
3) 每个 chunk 3 题：强制覆盖不同 Bloom（默认 B1/B3/B4 轮换）
4) 输出 TSV 增加 bloom_level、bloom_label 两列
"""

import csv
import os
import json
import time
from typing import List, Dict, Optional

from openai import OpenAI
from pipeline_config import STEP4_NODES_TSV, STEP4_EDGES_TSV, STEP2_SENT_TSV
from pipeline_config import STEP8_Q_TSV, PROMPT_PATH_BLOOM

NODES_TSV = str(STEP4_NODES_TSV)
EDGES_TSV = str(STEP4_EDGES_TSV)
SENT_TSV = str(STEP2_SENT_TSV)
OUTPUT_Q_TSV = str(STEP8_Q_TSV)

# ========== 路径配置 ==========

# NODES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_nodes_updated.tsv"
# EDGES_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step12_output\第一讲_KG_edges_updated.tsv"

# SENT_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step2_output\第一讲_句子列表.tsv"

# ✅ 改：使用新的 system prompt（见下方提示词内容）
# PROMPT_PATH = r"D:\Desktop\KG_allprocess\KG_code\prompt_bloom_same_knowledge.txt"
PROMPT_PATH = str(PROMPT_PATH_BLOOM)
# OUTPUT_Q_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step8_output\第一讲_MCQ_bloom.tsv"


# ========== LLM 配置 ==========

import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ========== LLM 配置 ==========

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key = config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

MODEL_NAME = "DeepSeek-R1"


# ========== 生成策略参数 ==========

MAX_QUESTIONS = 50          # 最多生成多少题
EDGES_PER_CHUNK = 5         # 每次给 LLM 的 fact 条数
QUESTIONS_PER_CHUNK = 6     # 每个 chunk 期望生成几道题（上限值）

# ✅ 每次 chunk 3 题的 Bloom 目标分布（强约束）
# 你也可以改成 ["B1","B2","B3"] 或 ["B2","B4","B5"] 等
BLOOM_PATTERN = ["B1", "B2", "B3", "B4", "B5", "B6"]

# ✅ Bloom 映射
BLOOM_LEVELS = {
    "B1": "记忆",
    "B2": "理解",
    "B3": "应用",
    "B4": "分析",
    "B5": "评价",
    "B6": "创造",
}


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
    （不依赖列名）
    """
    sentences: Dict[str, Dict] = {}
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


def build_fact_strings(
    nodes: Dict[str, Dict],
    edges: List[Dict],
    sentences: Dict[str, Dict],
) -> List[str]:
    """
    把 KG 的边 + 对应原句一起变成“可读事实字符串”，喂给 LLM。
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
    """把列表按固定大小切分成多个小块。"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def extract_json_array(text: str) -> Optional[str]:
    """
    从模型输出里尽量提取 JSON 数组部分：[ ... ]
    兼容 ```json ... ``` 或前后有杂讯。
    """
    if not text:
        return None

    t = text.strip()

    # 去掉最外层可能的 ``` 包裹
    if t.startswith("```"):
        # 可能是 ```json\n...\n```
        t = t.strip("`").strip()
        # 如果还包含 'json' 标记，去掉其前缀行
        if t.lower().startswith("json"):
            t = t[4:].strip()

    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return t[start:end + 1]


def bloom_targets_for_chunk(chunk_index_1based: int, n_questions: int) -> List[str]:
    """
    给第 chunk_index 个 chunk 分配 bloom 目标（强约束），按 BLOOM_PATTERN 轮换。
    默认每 chunk 3 题：
      chunk1: B1,B3,B4
      chunk2: B1,B3,B4
    你也可以扩展成随 chunk 变化的模式。
    """
    # 最简单：固定模式截取 n_questions
    pattern = BLOOM_PATTERN[:]
    if n_questions <= len(pattern):
        return pattern[:n_questions]

    # 若你设了 n_questions > pattern 长度，则循环补齐
    out = []
    while len(out) < n_questions:
        out.extend(pattern)
    return out[:n_questions]


# ========== LLM 调用：生成 MCQ（带 Bloom） ==========

def call_llm_for_mcq(fact_chunk: List[str], target_blooms: List[str]) -> List[Dict]:
    """
    调用 LLM：基于一个 fact_chunk 生成若干道单选题，并强制 Bloom 层级。
    返回 item 需包含：
      question/options/answer/bloom_level/bloom_label
    """
    n_questions = len(target_blooms)
    if not fact_chunk or n_questions <= 0:
        return []

    # 给 fact 编号
    facts_text = "\n".join(f"{idx + 1}. {f}" for idx, f in enumerate(fact_chunk))

    # 给 Bloom 目标编号，强制每题一个层级
    bloom_req_text = "\n".join(
        f"- 第{i+1}题 bloom_level 必须是 {b}（{BLOOM_LEVELS[b]}）"
        for i, b in enumerate(target_blooms)
        if b in BLOOM_LEVELS
    )

    user_prompt = f"""
下面是若干条来自法律知识图谱的“事实及其来源原句”：

{facts_text}

说明：
- 每条记录通常包含两行：
  - “事实：...” 行描述知识图谱中的实体关系
  - “来源原句：...” 行给出该事实在原始材料中的完整句子（若有）

请你【仅根据上述事实及原句】生成 {n_questions} 道中文法律单选题（MCQ），并满足以下要求：

1. 每道题必须包含字段：
   - "question"：题干（中文表述，可适当改写原句，但不能脱离原意）
   - "options"：包含四个元素的数组 ["A. ...", "B. ...", "C. ...", "D. ..."]
   - "answer"：正确选项的选项字母（只能是 "A"、"B"、"C" 或 "D"）
   - "bloom_level"：只能是 ["B1","B2","B3","B4","B5","B6"] 之一
   - "bloom_label"：只能是 ["记忆","理解","应用","分析","评价","创造"] 之一

2. 题目内容和选项必须能够从【事实 + 原句】中推导出来，不允许引入材料中没有的信息（包括常识性法律知识、背景法条、推测性结论）。
3. 四个选项都要合理、有一定迷惑性，但不能明显错误或与原句矛盾。
4. 每道题只能有一个唯一正确答案，不要出现多选或模糊不清的情况。
5. 不要输出题目解析或任何解释。
6. 最终输出必须是一个 JSON 数组，例如：
   [
     {{"question":"...","options":["A. ...","B. ...","C. ...","D. ..."],"answer":"B","bloom_level":"B2","bloom_label":"理解"}},
     ...
   ]
7. 不要在 JSON 外输出任何额外文字（不要 markdown、不要说明，只要 JSON）。

【布鲁姆层级约束（必须严格遵守）】
- B1 记忆：考察对原句/事实的直接回忆或识别。
- B2 理解：考察释义、同义改写、概括归纳（仍完全忠于材料）。
- B3 应用：将材料中表达的关系/规则用于“材料内部等价表述”的判断；不得引入新主体/新条件/新结论。
- B4 分析：拆分要素、辨析关系（主体/客体/关系类型/条件-结论结构），仍不得引入材料外信息。
- B5 评价：仅在材料内部做一致性/贴合度判断；不得用材料外标准或价值判断依据。
- B6 创造：只能做“基于材料的重组表达/规范化表述/抽象概括”，不得新增材料外事实、条件、主体或结论。

【本次 Bloom 强制分配】
{bloom_req_text}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw = (response.choices[0].message.content or "").strip()
    json_text = extract_json_array(raw)
    if not json_text:
        print("⚠ 未能从模型输出中提取 JSON 数组。输出片段：", raw[:200], "...")
        return []

    try:
        data = json.loads(json_text)
    except Exception as e:
        print("⚠ 解析 JSON 失败：", e)
        print("原始 JSON 片段：", json_text[:300], "...")
        return []

    if not isinstance(data, list):
        return []

    mcqs: List[Dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        q = str(item.get("question", "")).strip()
        options = item.get("options", [])
        answer = str(item.get("answer", "")).strip()

        bloom_level = str(item.get("bloom_level", "")).strip()
        bloom_label = str(item.get("bloom_label", "")).strip()

        if not q or not isinstance(options, list) or len(options) != 4:
            continue
        options = [str(opt).strip() for opt in options]

        if answer not in ("A", "B", "C", "D"):
            continue

        if bloom_level not in BLOOM_LEVELS:
            continue

        # 自动补全/校验 bloom_label
        expected_label = BLOOM_LEVELS[bloom_level]
        if bloom_label:
            if bloom_label != expected_label:
                continue
        else:
            bloom_label = expected_label

        # ✅ 强制检查 bloom 是否匹配目标分配
        if i < len(target_blooms):
            if bloom_level != target_blooms[i]:
                continue

        mcqs.append({
            "question": q,
            "options": options,
            "answer": answer,
            "bloom_level": bloom_level,
            "bloom_label": bloom_label,
        })

    # 最多返回 n_questions
    return mcqs[:n_questions]


# ========== 主逻辑：分块生成 MCQ + 进度显示 ==========

def generate_mcq_with_llm(
    nodes: Dict[str, Dict],
    edges: List[Dict],
    sentences: Dict[str, Dict],
) -> List[Dict]:
    """
    分块喂 facts（KG 边 + 原句），调用 LLM 生成多道单选题（带 Bloom）。
    """
    facts = build_fact_strings(nodes, edges, sentences)
    chunks = chunk_list(facts, EDGES_PER_CHUNK)

    all_mcq_rows: List[Dict] = []
    q_counter = 1
    total_chunks = len(chunks)
    avg_call_time: List[float] = []

    print(f"\n🔄 共 {total_chunks} 个 fact chunk，将生成最多 {MAX_QUESTIONS} 道题（带 Bloom 标注）\n")

    for chunk_idx, fact_chunk in enumerate(chunks, start=1):
        if len(all_mcq_rows) >= MAX_QUESTIONS:
            break

        remain = MAX_QUESTIONS - len(all_mcq_rows)
        n_q = min(QUESTIONS_PER_CHUNK, remain)

        target_blooms = bloom_targets_for_chunk(chunk_idx, n_q)

        print(f"\n📌 Chunk {chunk_idx}/{total_chunks}: 尝试生成 {n_q} 道题 | Bloom 目标：{target_blooms}")
        print("   🤖 调用 LLM 中……")

        start_time = time.time()
        mcqs = call_llm_for_mcq(fact_chunk, target_blooms)
        cost = time.time() - start_time
        avg_call_time.append(cost)

        print(f"   ✅ LLM 返回（耗时 {cost:.2f} 秒），通过校验题数：{len(mcqs)}")

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
                "bloom_level": item.get("bloom_level", ""),
                "bloom_label": item.get("bloom_label", ""),
            }
            all_mcq_rows.append(row)
            q_counter += 1

        # ETA
        progress = len(all_mcq_rows)
        if avg_call_time and progress > 0:
            avg_t = sum(avg_call_time) / len(avg_call_time)
            remain_calls = max((MAX_QUESTIONS - progress) / max(QUESTIONS_PER_CHUNK, 1), 0)
            eta = remain_calls * avg_t
            print(f"   📊 进度：已生成 {progress}/{MAX_QUESTIONS} 道题 | 预估剩余时间约：{eta:.1f} 秒")

    print("\n🎉 题目生成完成！")
    return all_mcq_rows


# ========== 保存 TSV ==========

def save_mcq(rows: List[Dict], path: str):
    """保存 TSV：qid, question, option_a, option_b, option_c, option_d, answer, bloom_level, bloom_label"""
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
        "bloom_level",
        "bloom_label",
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
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError("请检查 prompt 路径是否正确（PROMPT_PATH）")

    nodes = load_nodes(NODES_TSV)
    edges = load_edges(EDGES_TSV)
    sentences = load_sentences(SENT_TSV)

    print(f"📄 已加载节点数：{len(nodes)}，边数：{len(edges)}，句子数：{len(sentences)}")

    mcq_rows = generate_mcq_with_llm(nodes, edges, sentences)
    save_mcq(mcq_rows, OUTPUT_Q_TSV)


if __name__ == "__main__":
    main()
