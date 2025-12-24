"""
Step10 — 自动提升 MCQ 题目质量（基于 Step9 评估结果 + LLM 自动改写）

读取：STEP9_EVAL_TSV（TSV，来自 Step9 输出）
写出：STEP10_Q_REVISED_TSV（TSV，包含原题+改写题+原因）

✅ 适配你当前 Step9 输出字段（百分制 0~100）：
- qid, question, option_a, option_b, option_c, option_d, answer
- A_entity_relation_coverage_score, B_coherence_score, C_entity_alignment_score, D_relation_correctness_score, Q_total
- 其他字段保留（如 ent_cnt/rel_cnt/entities_matched）

改写规则：
- 仅当 Q_total < THRESHOLD_Q（百分制）才调用 LLM 改写
- 必须保持正确答案字母不变（answer=A/B/C/D）
- 输出 JSON 再写入 TSV
"""

import csv
import json
import re
import time
from typing import Dict, List, Any, Optional

import yaml
from openai import OpenAI

from pipeline_config import STEP9_EVAL_TSV, STEP10_Q_REVISED_TSV


# =============== 配置区域 ===============

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Step9 输出的评估结果 TSV（输入）
INPUT_EVAL_TSV = str(STEP9_EVAL_TSV)

# 自动改写后的题目 TSV（输出）
OUTPUT_Q_REVISED_TSV = str(STEP10_Q_REVISED_TSV)

# ✅ 你的 Step9 Q_total 是 0~100（百分制），这里也用百分制阈值
THRESHOLD_Q = 85.0  # 你可改：例如 90.0

# Gitee + DeepSeek 客户端（保持你原来的写法）
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

MODEL_NAME = "DeepSeek-R1"  # 你在 Gitee 上真实可用的模型名


# =============== 工具函数 ===============

def load_mcq_with_scores(path: str) -> List[Dict[str, str]]:
    """读取 Step9 的评估结果 TSV"""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_json(text: str) -> Dict[str, Any]:
    """
    尽量从模型输出中提取 JSON：
    - 直接 json.loads
    - ```json ... ```
    - 从第一个 { 到最后一个 }
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        candidate = text[i:j + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"无法解析 JSON，模型输出为：{text[:800]}")


def call_llm_for_improvement(row: Dict[str, str]) -> Dict[str, Any]:
    """
    调用 LLM 对一道题进行“自动改进”。
    输入：原始 MCQ + Step9 评估分数（帮助模型定位问题）
    输出：包含改写后的 question/options/answer/revision_reason 的 dict
    """

    qid = row.get("qid", "")
    question = (row.get("question", "") or "").strip()
    option_a = (row.get("option_a", "") or "").strip()
    option_b = (row.get("option_b", "") or "").strip()
    option_c = (row.get("option_c", "") or "").strip()
    option_d = (row.get("option_d", "") or "").strip()
    answer = (row.get("answer", "") or "").strip().upper()

    # Step9 分数（百分制）
    A = row.get("A_entity_relation_coverage_score", "")
    B = row.get("B_coherence_score", "")
    C = row.get("C_entity_alignment_score", "")
    D = row.get("D_relation_correctness_score", "")
    Q = row.get("Q_total", "")

    # 你 Step9 里可能还有 context/fact（没有也没关系）
    context = (row.get("context", "") or "").strip()
    fact = (row.get("fact", "") or "").strip()

    extra_context = ""
    if context:
        extra_context += f"\n【上下文】\n{context}\n"
    if fact:
        extra_context += f"\n【KG事实】\n{fact}\n"

    prompt = f"""
你是一个“选择题质量改写器”，目标是提升题目质量，但必须严格遵守：

【硬约束】
1) 题目仍然是单选题；
2) 正确答案字母必须保持为：{answer} （不可更换为其他选项）；
3) 不要引入题干/上下文未给出的外部事实；如无法确定事实，请改成不依赖外部事实也能判断的表述。

【软目标（尽量做到）】
- 提升语义连贯性；
- 题干与选项实体对齐（避免出现题干说 A，选项在说 B）；
- 如果原题“缺乏关系考察”，尽量让题干更明确地体现一个关系（例如 属于/导致/组成/条件/对比/来源/定义 等），但仍不引入外部事实；
- 干扰项更像、更迷惑，但不能与正确答案等价。

【题目ID】{qid}

【原题干】
{question}

【原始选项】
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

【原始正确答案】（不要改字母）：{answer}

【Step9 评估分数（百分制 0~100）】
- 覆盖度 A_entity_relation_coverage_score: {A}
- 连贯性 B_coherence_score: {B}
- 实体对齐 C_entity_alignment_score: {C}
- 关系正确性 D_relation_correctness_score: {D}
- 总分 Q_total: {Q}
{extra_context}

请只输出下面 JSON（不要输出任何额外文字）：

{{
  "question": "改写后的题干",
  "option_a": "改写后的 A 选项（不要带 'A.' 前缀也可以）",
  "option_b": "改写后的 B 选项",
  "option_c": "改写后的 C 选项",
  "option_d": "改写后的 D 选项",
  "answer": "{answer}",
  "revision_reason": "你做了哪些修改以及原因（尽量对应分数低的点）"
}}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content
    obj = _extract_json(content)

    # 基础字段兜底
    obj["answer"] = (obj.get("answer", answer) or answer).strip().upper()
    if obj["answer"] != answer:
        # 强制保证答案字母不变
        obj["answer"] = answer

    # 为空则回退原文
    obj["question"] = (obj.get("question", "") or "").strip() or question
    obj["option_a"] = (obj.get("option_a", "") or "").strip() or option_a
    obj["option_b"] = (obj.get("option_b", "") or "").strip() or option_b
    obj["option_c"] = (obj.get("option_c", "") or "").strip() or option_c
    obj["option_d"] = (obj.get("option_d", "") or "").strip() or option_d
    obj["revision_reason"] = (obj.get("revision_reason", "") or "").strip()

    return obj


def save_revised(rows: List[Dict[str, Any]], path: str) -> None:
    """保存改写结果 TSV"""
    if not rows:
        raise ValueError("没有要写出的 rows")

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =============== 主流程 ===============

def main():
    rows = load_mcq_with_scores(INPUT_EVAL_TSV)
    if not rows:
        print(f"输入为空：{INPUT_EVAL_TSV}")
        return

    revised_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        qid = row.get("qid", "")
        question = (row.get("question", "") or "").strip()
        option_a = (row.get("option_a", "") or "").strip()
        option_b = (row.get("option_b", "") or "").strip()
        option_c = (row.get("option_c", "") or "").strip()
        option_d = (row.get("option_d", "") or "").strip()
        answer = (row.get("answer", "") or "").strip().upper()

        # ✅ 百分制 Q_total
        Q_before = parse_float(row.get("Q_total", 0.0), default=0.0)

        print("=" * 60)
        print(f"[{idx}/{len(rows)}] QID={qid}  Q_total={Q_before:.2f}")

        if Q_before >= THRESHOLD_Q:
            print("分数已达阈值，保持原题。")
            revised_rows.append(
                {
                    "qid": qid,
                    "Q_total_before": f"{Q_before:.2f}",
                    # 原题
                    "original_question": question,
                    "original_option_a": option_a,
                    "original_option_b": option_b,
                    "original_option_c": option_c,
                    "original_option_d": option_d,
                    "original_answer": answer,
                    # 改写题（同原题）
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
            continue

        # 需要改写
        print("低于阈值，开始调用 LLM 改写...")

        improved: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None

        # 简单重试 2 次
        for attempt in range(3):
            try:
                improved = call_llm_for_improvement(row)
                break
            except Exception as e:
                last_err = e
                print(f"LLM 改写失败 attempt={attempt+1}/3: {e}")
                time.sleep(1.0)

        if improved is None:
            print("改写失败，回退原题。")
            revised_rows.append(
                {
                    "qid": qid,
                    "Q_total_before": f"{Q_before:.2f}",
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
                    "revision_reason": f"LLM_FAILED: {last_err}",
                }
            )
            continue

        changed = "yes" if any(
            [
                improved.get("question", "").strip() != question,
                improved.get("option_a", "").strip() != option_a,
                improved.get("option_b", "").strip() != option_b,
                improved.get("option_c", "").strip() != option_c,
                improved.get("option_d", "").strip() != option_d,
            ]
        ) else "no"

        revised_rows.append(
            {
                "qid": qid,
                "Q_total_before": f"{Q_before:.2f}",
                "original_question": question,
                "original_option_a": option_a,
                "original_option_b": option_b,
                "original_option_c": option_c,
                "original_option_d": option_d,
                "original_answer": answer,
                "revised_question": improved.get("question", question),
                "revised_option_a": improved.get("option_a", option_a),
                "revised_option_b": improved.get("option_b", option_b),
                "revised_option_c": improved.get("option_c", option_c),
                "revised_option_d": improved.get("option_d", option_d),
                "revised_answer": answer,  # 强制保持不变
                "changed": changed,
                "revision_reason": improved.get("revision_reason", ""),
            }
        )

    save_revised(revised_rows, OUTPUT_Q_REVISED_TSV)
    print(f"\n✅ Step10 完成，输出已保存：{OUTPUT_Q_REVISED_TSV}")


if __name__ == "__main__":
    main()
