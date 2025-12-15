"""
Step10 — 自动提升 MCQ 题目质量（基于 Step9 评估结果 + LLM 自动改写）

流程：
1. 读取 Step9_evaluate_questions.py 的输出 TSV：
   必需字段：
     - qid
     - question
     - option_a, option_b, option_c, option_d
     - answer  （A/B/C/D）
     - Q_total （综合质量评分，0~1）
   可选字段：
     - context （原句）
     - fact    （KG 事实）

2. 对 Q_total < THRESHOLD 的题目调用 LLM 进行“自动改写”，目标：
   - 提升：faithfulness, coverage, distractor quality, fluency
   - 不改变考点：正确选项（answer 对应的含义）保持正确
   - 可以重写题干和干扰项，使题目更清晰、有迷惑性且忠于上下文/事实

3. 生成一个新的 TSV，包含原题与改写后的题：
   - qid
   - original_question, original_option_a, ..., original_answer
   - revised_question, revised_option_a, ..., revised_answer
   - changed (yes/no)
   - revision_reason
   - Q_total_before
"""

import csv
import os
import json
from typing import Dict, List, Any

from openai import OpenAI

# =============== 配置区域 ===============

# Step9 输出的评估结果 TSV（输入）
INPUT_EVAL_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step9_output\第一讲_MCQ_eval.tsv"

# 自动改写后的题目 TSV（输出）
OUTPUT_Q_REVISED_TSV = r"D:\Desktop\KG_allprocess\KG_files\Output_files\Step10_output\第一讲_MCQ_auto_revised.tsv"

# 综合质量评分阈值：低于此值的题目会尝试自动改写
THRESHOLD_Q = 0.90

# Gitee + DeepSeek 客户端
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="DUQFR61KA8QLDVEQPGJKBXYSL2DXMPST1FM98Y1L",
    default_headers={"X-Failover-Enabled":"true"},
)

CHAT_MODEL = "DeepSeek-R1"  # 按你在 Gitee 上真实可用的聊天模型名称修改


# =============== 工具函数 ===============

def load_mcq_with_scores(path: str) -> List[Dict[str, str]]:
    """读取 Step9 的评估结果 TSV"""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def call_llm_for_improvement(row: Dict[str, str]) -> Dict[str, Any]:
    """
    调用 LLM 对一道题进行“自动改进”。
    输入：原始 MCQ + 评估分数（可辅助 LLM 理解哪里不好）
    输出：包含改写后的 question / options / answer / revision_reason 的 dict
    """

    qid = row.get("qid", "")
    question = row.get("question", "").strip()
    option_a = row.get("option_a", "").strip()
    option_b = row.get("option_b", "").strip()
    option_c = row.get("option_c", "").strip()
    option_d = row.get("option_d", "").strip()
    answer = row.get("answer", "").strip().upper()

    context = row.get("context", "").strip() if "context" in row else ""
    fact = row.get("fact", "").strip() if "fact" in row else ""

    F_before = parse_float(row.get("F_faithfulness", "0"))
    D_before = parse_float(row.get("D_distractor", "0"))
    C_before = parse_float(row.get("C_coverage", "0"))
    L_before = parse_float(row.get("L_fluency", "0"))
    Q_before = parse_float(row.get("Q_total", "0"))

    extra_context = ""
    if context:
        extra_context += f"\n【原始上下文】\n{context}\n"
    if fact:
        extra_context += f"\n【知识图谱事实】\n{fact}\n"

    score_info = (
        f"faithfulness(忠实性)={F_before:.2f}, "
        f"distractor(干扰项质量)={D_before:.2f}, "
        f"coverage(覆盖度)={C_before:.2f}, "
        f"fluency(流畅度)={L_before:.2f}, "
        f"Q_total(综合)={Q_before:.2f}"
    )

    user_prompt = f"""
你是一名严格的法律考试命题专家，现在要对一题自动生成的单项选择题进行质量改进。

请参考以下四个维度：
1. faithfulness：题目是否忠实于原始上下文/知识事实，是否不存在明显错误或捏造信息。
2. coverage：题目是否覆盖了应考查的核心语义与知识点，而不是偏题或遗漏重要信息。
3. distractor：干扰项是否设计合理、有一定迷惑性，不是明显错误或过于相似。
4. fluency：题干和选项的中文表达是否流畅、清晰、符合正式法律/教材语言。

下面是原始题目及其自动评估分数（0~1）：
QID: {qid}
综合评分及各维度评分：{score_info}

【原始题干】
{question}

【原始选项】
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

【原始正确答案】（选项字母，不要修改）：{answer}

{extra_context}

请你在【不改变考查核心知识点和正确选项含义】的前提下，对题干和选项进行改写，以提升上述各维度的质量。
具体要求：
- 正确答案对应的选项字母（{answer}）必须保持不变，即改写后仍然是该选项为唯一正确答案。
- 可以改写题干，使其更加清晰、明确、忠实于上下文/事实。
- 可以改写干扰项，使其更有迷惑性，但不能与正确答案等价或明显错误。
- 如原题严重不合理，可以对整题结构进行适度调整，但要保证仍然是单项选择题。

请使用如下 JSON 格式给出改写后的结果（不要添加任何其他说明文字）：

{{
  "question": "改写后的题干",
  "option_a": "改写后的 A 选项",
  "option_b": "改写后的 B 选项",
  "option_c": "改写后的 C 选项",
  "option_d": "改写后的 D 选项",
  "answer": "{answer}",
  "revision_reason": "简要说明你做了哪些改进（例如增强干扰项迷惑性、消除歧义、贴近原文语义等）。"
}}
""".strip()

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一名细致、严谨的中文法律考试命题与审题专家，擅长改写和优化考题。",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = resp.choices[0].message.content or ""
    # 尝试解析 JSON
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            js = content[start : end + 1]
        else:
            js = content

        data = json.loads(js)

        # 确保必要字段存在
        out = {
            "question": data.get("question", "").strip() or question,
            "option_a": data.get("option_a", "").strip() or option_a,
            "option_b": data.get("option_b", "").strip() or option_b,
            "option_c": data.get("option_c", "").strip() or option_c,
            "option_d": data.get("option_d", "").strip() or option_d,
            # answer 必须保持原来的字母
            "answer": answer,
            "revision_reason": data.get("revision_reason", "").strip(),
        }
        return out

    except Exception as e:
        print(f"⚠ 解析 LLM 返回 JSON 失败，QID={qid}，错误：{e}")
        # 解析失败就返回原题，标记为未修改
        return {
            "question": question,
            "option_a": option_a,
            "option_b": option_b,
            "option_c": option_c,
            "option_d": option_d,
            "answer": answer,
            "revision_reason": "LLM 输出解析失败，未进行改写。",
        }


def save_revised(rows: List[Dict[str, Any]], out_path: str):
    """保存自动改写后的题目 TSV"""
    if not rows:
        print("没有任何题目可写出。")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "qid",
        "Q_total_before",
        "original_question",
        "original_option_a",
        "original_option_b",
        "original_option_c",
        "original_option_d",
        "original_answer",
        "revised_question",
        "revised_option_a",
        "revised_option_b",
        "revised_option_c",
        "revised_option_d",
        "revised_answer",
        "changed",
        "revision_reason",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ 自动改写结果已保存到：{out_path}")
    print(f"   共 {len(rows)} 道题目。")


def main():
    if not os.path.exists(INPUT_EVAL_TSV):
        raise FileNotFoundError(f"找不到评估结果文件: {INPUT_EVAL_TSV}")

    rows = load_mcq_with_scores(INPUT_EVAL_TSV)
    print(f"共读取到 {len(rows)} 道题目（含评估结果）。")

    revised_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        qid = row.get("qid", f"q{idx:04d}")
        Q_before = parse_float(row.get("Q_total", "0"))

        question = row.get("question", "").strip()
        option_a = row.get("option_a", "").strip()
        option_b = row.get("option_b", "").strip()
        option_c = row.get("option_c", "").strip()
        option_d = row.get("option_d", "").strip()
        answer = row.get("answer", "").strip().upper()

        print("=" * 60)
        print(f"[{idx}/{len(rows)}] QID={qid}  Q_total={Q_before:.4f}")

        if Q_before >= THRESHOLD_Q:
            # 评分已达标，不做改写
            print("分数已达阈值，保持原题。")
            revised_rows.append(
                {
                    "qid": qid,
                    "Q_total_before": f"{Q_before:.4f}",
                    "original_question": question,
                    "original_option_a": option_a,
                    "original_option_b": option_b,
                    "original_option_c": option_c,
                    "original_option_d": option_d,
                    "original_answer": answer,
                    "revised_question": question,
                    "revised_option_a": option_a,
                    "revised_option_b": option_b,
                    "revised_option_c": option_c,
                    "revised_option_d": option_d,
                    "revised_answer": answer,
                    "changed": "no",
                    "revision_reason": "",
                }
            )
        else:
            print("分数较低，调用 LLM 进行自动改写...")
            improved = call_llm_for_improvement(row)

            revised_rows.append(
                {
                    "qid": qid,
                    "Q_total_before": f"{Q_before:.4f}",
                    "original_question": question,
                    "original_option_a": option_a,
                    "original_option_b": option_b,
                    "original_option_c": option_c,
                    "original_option_d": option_d,
                    "original_answer": answer,
                    "revised_question": improved["question"],
                    "revised_option_a": improved["option_a"],
                    "revised_option_b": improved["option_b"],
                    "revised_option_c": improved["option_c"],
                    "revised_option_d": improved["option_d"],
                    "revised_answer": improved["answer"],
                    "changed": "yes" if improved["question"] != question or any(
                        [
                            improved["option_a"] != option_a,
                            improved["option_b"] != option_b,
                            improved["option_c"] != option_c,
                            improved["option_d"] != option_d,
                        ]
                    ) else "no",
                    "revision_reason": improved.get("revision_reason", ""),
                }
            )

    save_revised(revised_rows, OUTPUT_Q_REVISED_TSV)


if __name__ == "__main__":
    main()
