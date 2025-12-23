"""
Step9 — 纯 LLM 版本的 MCQ 质量评估（不依赖 embeddings 接口）

功能：
1. 从 Step8 输出 TSV 中读取题目：
   必需列：qid, question, option_a, option_b, option_c, option_d, answer
   可选列：context, fact（如果以后你在 Step8 里加上，也会被利用）

2. 对每道题调用一次 LLM（DeepSeek-R1 等），请它按 4 个维度打分：
   - faithfulness: 忠实性 / 逻辑自洽（1~5）
   - distractor: 干扰项质量（1~5）
   - coverage: 语义覆盖度 / 信息完整性（1~5）
   - fluency: 语言流畅度（1~5）

3. 将 1~5 分归一化到 0~1，并按公式计算总分：
   F = faithfulness / 5
   D = distractor / 5
   C = coverage / 5
   L = fluency / 5

   Q = 0.30 * F + 0.30 * D + 0.25 * C + 0.15 * L

4. 输出新的 TSV，附加列：
   F_faithfulness, D_distractor, C_coverage, L_fluency, Q_total
"""

import csv
import os
from typing import Dict, List, Any

from openai import OpenAI
from pipeline_config import STEP9_EVAL_TSV, STEP8_Q_TSV

# ================== 配置区域 ==================

# Step8 输出的题目 TSV 路径（请按你的实际路径修改）
# INPUT_Q_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step8_output\第一讲_MCQ.tsv"
INPUT_Q_TSV = str(STEP8_Q_TSV)
# Step9 输出的评估结果 TSV 路径
# OUTPUT_EVAL_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step9_output\第一讲_MCQ_eval.tsv"
OUTPUT_EVAL_TSV = str(STEP9_EVAL_TSV)

# Gitee + DeepSeek 客户端（和你 Step8 保持一致）
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="DUQFR61KA8QLDVEQPGJKBXYSL2DXMPST1FM98Y1L",
    default_headers={"X-Failover-Enabled":"true"},
)

# LLM 模型名（用于评分）
CHAT_MODEL = "DeepSeek-R1"  # 按你在 Gitee 上实际可用的模型名来


# ================== 工具函数 ==================

def load_mcq(path: str) -> List[Dict[str, str]]:
    """读取 Step8 生成的 MCQ TSV"""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def parse_scores_from_llm(content: str) -> Dict[str, float]:
    """
    从 LLM 返回的文本中解析出四个 1~5 的整数评分：
    faithfulness, distractor, coverage, fluency

    期望格式类似：
    {
      "faithfulness": 4,
      "distractor": 3,
      "coverage": 5,
      "fluency": 4
    }

    但为了鲁棒性，这里用很宽松的解析方式：
    - 先尝试按 JSON 解析；
    - 不行的话，就在文本里找这几个字段名后面的数字。
    """
    import json
    result = {
        "faithfulness": 3,
        "distractor": 3,
        "coverage": 3,
        "fluency": 3,
    }

    text = content.strip()
    # 先尝试 JSON
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            js = text[start:end+1]
            data = json.loads(js)
            for key in result.keys():
                if key in data:
                    v = int(data[key])
                    if 1 <= v <= 5:
                        result[key] = v
            return result
    except Exception:
        pass

    # fallback：简单扫一遍文本，找数字
    # 格式示例：
    # faithfulness: 4
    # distractor: 3
    # coverage: 5
    # fluency: 4
    lowered = text.lower()
    for key in result.keys():
        idx = lowered.find(key)
        if idx != -1:
            # 从该位置往后找 1~5 的数字
            tail = lowered[idx: idx + 50]
            digits = [ch for ch in tail if ch.isdigit()]
            if digits:
                v = int(digits[0])
                if 1 <= v <= 5:
                    result[key] = v

    return result


def llm_score_mcq(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    answer: str,
    context: str = "",
    fact: str = "",
) -> Dict[str, float]:
    """
    调用 LLM 对一整道题做 4 个维度的评分：
    faithfulness, distractor, coverage, fluency  (1~5)
    返回 dict，数值还是 1~5（后面再归一化）
    """

    # 说明文字中，把 context/fact 作为可选信息传给模型
    extra_context = ""
    if context:
        extra_context += f"\n【原始上下文】\n{context}\n"
    if fact:
        extra_context += f"\n【知识图谱事实】\n{fact}\n"

    user_prompt = f"""
你是一名严谨的法律考试命题专家。请从下面四个维度对一道多项选择题进行评分，每个维度 1~5 分，5 分为最好：

1. faithfulness：题目与给定的原始上下文/知识事实是否一致、是否逻辑自洽；若未提供上下文，则判断题干内部是否自洽、是否不存在明显矛盾或错误。
2. distractor：干扰项质量是否较高，例如选项内容看起来都合理，不明显错误，也不过分重复，错误项不应一眼就被排除。
3. coverage：题目是否覆盖了关键信息，考查点是否完整、明确；若提供了上下文/知识事实，则看题目是否覆盖其核心语义。
4. fluency：题干语言是否通顺、语法正确、表述自然，符合正式中文法律/教材风格。

请根据题目整体情况，为这四个维度分别打 1~5 分（可以使用整数即可）。

题目如下：

【题干】
{question}

【选项】
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

【正确答案】（仅供你评估用，不需要质疑或修改选项）：{answer}

{extra_context}

请严格按照以下 JSON 格式输出，不要添加任何其它文字：

{{
  "faithfulness": 分数(1-5),
  "distractor": 分数(1-5),
  "coverage": 分数(1-5),
  "fluency": 分数(1-5)
}}
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一名细致严格的中文法律考试命题与审题专家。"},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or ""
    scores_1_to_5 = parse_scores_from_llm(content)
    return scores_1_to_5


def evaluate_mcq_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    对每道题进行评估，附加新字段：
    - F_faithfulness, D_distractor, C_coverage, L_fluency, Q_total
    所有分数是 0~1 或 0~1 再乘权重后的总分。
    """
    evaluated: List[Dict[str, Any]] = []

    has_context = rows and ("context" in rows[0])
    has_fact = rows and ("fact" in rows[0])

    for idx, r in enumerate(rows, start=1):
        qid = r.get("qid", f"q{idx:04d}")
        question = r.get("question", "").strip()
        option_a = r.get("option_a", "").strip()
        option_b = r.get("option_b", "").strip()
        option_c = r.get("option_c", "").strip()
        option_d = r.get("option_d", "").strip()
        answer = r.get("answer", "").strip().upper()

        context = r.get("context", "").strip() if has_context else ""
        fact = r.get("fact", "").strip() if has_fact else ""

        print(f"评估题目 {qid} ...")

        scores_raw = llm_score_mcq(
            question=question,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
            answer=answer,
            context=context,
            fact=fact,
        )

        # 1~5 -> 0~1
        F = max(1, min(5, int(scores_raw["faithfulness"]))) / 5.0
        D = max(1, min(5, int(scores_raw["distractor"]))) / 5.0
        C = max(1, min(5, int(scores_raw["coverage"]))) / 5.0
        L = max(1, min(5, int(scores_raw["fluency"]))) / 5.0

        Q = 0.30 * F + 0.30 * D + 0.25 * C + 0.15 * L

        r_out = dict(r)
        r_out["F_faithfulness"] = f"{F:.4f}"
        r_out["D_distractor"] = f"{D:.4f}"
        r_out["C_coverage"] = f"{C:.4f}"
        r_out["L_fluency"] = f"{L:.4f}"
        r_out["Q_total"] = f"{Q:.4f}"

        evaluated.append(r_out)

    return evaluated


def save_eval_rows(rows: List[Dict[str, Any]], path: str):
    """保存带评估结果的 TSV"""
    if not rows:
        print("没有数据可写出")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ 评估结果已保存到：{path}（共 {len(rows)} 题）")


def main():
    if not os.path.exists(INPUT_Q_TSV):
        raise FileNotFoundError(f"找不到输入题目文件：{INPUT_Q_TSV}")

    rows = load_mcq(INPUT_Q_TSV)
    print(f"共读取到 {len(rows)} 道题目。")

    eval_rows = evaluate_mcq_rows(rows)
    save_eval_rows(eval_rows, OUTPUT_EVAL_TSV)


if __name__ == "__main__":
    main()
