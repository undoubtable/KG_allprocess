import os
import json
import yaml
import csv
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from pipeline_config import STEP2_SENT_TSV, STEP35_TRUTH_ENT_TSV

# ===================== 路径配置 =====================
sent_tsv_path = str(STEP2_SENT_TSV)
truth_entity_tsv_path = str(STEP35_TRUTH_ENT_TSV)

# ===================== LLM 配置 =====================
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key=config["api_key"],
    default_headers={"X-Failover-Enabled": "true"},
)

CANDIDATE_MODEL = "DeepSeek-V3"
VERIFY_MODEL = "DeepSeek-R1"  # 若想更快先跑通：可以改成 DeepSeek-V3

# ===================== 参数控制（推荐这样设） =====================
BATCH_SIZE = 10               # ✅ 不要 50，容易超时
MAX_CAND_PER_SENT = 12
MAX_TRUTH_PER_SENT = 6
DEFAULT_CONF_TRUTH = 0.95

# 候选太少就不走 R1（省时间）
ENABLE_SKIP_VERIFY = True
SKIP_VERIFY_IF_CAND_LEQ = 1

# 打印控制：只打印进度，不打印样例
PRINT_EVERY_N_BATCH = 1

# 请求重试
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 1.2

# ===================== 清洗/过滤配置 =====================
LIST_SEPS = ("、", "，", ",", "；", ";", "/", "／")
CH_NUMERIC_CHARS = set("一二三四五六七八九十百千万零〇0１２３４５６７８９0123456789")
BAD_GENERIC = {
    "行为","规定","情况","方面","问题","过程","内容","方式","结果","因素","原则","要求",
    "对象","责任","制度","标准","措施","情形","目的","性质","概念","关系","依据","条件",
    "范围","程度","方法","意见","决定","通知","公告","材料","证据","事实","理由","结论",
}
BAD_SUFFIX = ("方面","问题","情况","过程","内容","方式","结果","因素","原则","要求","制度","关系","标准","措施","情形")
MAX_MENTION_LEN = 20
MIN_LEN = 2


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
            print(f"⚠️ LLM请求失败（{model}）attempt {attempt}/{MAX_RETRIES}: {e} -> sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM 请求失败（重试仍失败）：{last_err}")


def load_sentences(tsv_path: str) -> List[Dict[str, Any]]:
    sents = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, page_no, text = parts[0], int(parts[1]), parts[2]
            sents.append({"sentence_id": sid, "page_no": page_no, "text": text})
    return sents


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


def clean_mention(m: str) -> str:
    m = re.sub(r"\s+", "", m).strip()
    m = m.strip("《》「」【】[]()（）\"'“”‘’")
    if len(m) > 3 and m[0] in ("和", "的"):
        m = m[1:]
    if m.startswith("刑法修正案(") and not (m.endswith(")") or m.endswith("）")):
        m += ")"
    if m.startswith("刑法修正案（") and not (m.endswith(")") or m.endswith("）")):
        m += "）"
    return m


def is_pure_numeric(m: str) -> bool:
    return bool(m) and all(ch in CH_NUMERIC_CHARS for ch in m)


def is_bad(m: str) -> bool:
    if len(m) < MIN_LEN or len(m) > MAX_MENTION_LEN:
        return True
    if is_pure_numeric(m):
        return True
    if m in BAD_GENERIC:
        return True
    if m.endswith(BAD_SUFFIX) and len(m) <= 6:
        return True
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", m):
        return True
    return False


def split_list(m: str) -> List[str]:
    if not any(sep in m for sep in LIST_SEPS):
        return [m]
    parts = [m]
    for sep in LIST_SEPS:
        new_parts = []
        for p in parts:
            new_parts.extend(p.split(sep))
        parts = new_parts
    parts = [clean_mention(p) for p in parts if p]
    parts = [p for p in parts if p and not is_bad(p)]
    # 去重保序
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out if out else [m]


def find_span(text: str, m: str) -> Optional[Tuple[int, int]]:
    start = text.find(m)
    if start == -1:
        return None
    return start, start + len(m)


def load_done_sentence_ids(path: str) -> set:
    """
    断点续跑：读取已写入的 truth_entity_tsv，返回已处理的 sentence_id 集合
    """
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("sentence_id")
            if sid:
                done.add(sid)
    return done


def append_entities_tsv(rows: List[Dict[str, Any]], path: str) -> None:
    """
    追加写入 TSV（若文件不存在则写 header）
    """
    need_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8", newline="") as f:
        if need_header:
            f.write("entity_id\tsentence_id\tpage_no\tmention\tstart_char\tend_char\tent_type\tconfidence\n")
        for r in rows:
            f.write(
                f"{r['entity_id']}\t{r['sentence_id']}\t{r['page_no']}\t{r['mention']}\t"
                f"{r['start_char']}\t{r['end_char']}\t{r['ent_type']}\t{r['confidence']:.4f}\n"
            )


def llm_candidate_entities_batch(batch: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    system = (
        "你是中文信息抽取助手。请对多个句子分别抽取实体。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"items\":[{\"sentence_id\":\"...\",\"entities\":[{\"mention\":\"...\",\"ent_type\":\"...\"}, ...]}, ...]}\n"
        "规则：\n"
        "1) mention 必须是对应句子中的连续子串，原文复制。\n"
        "2) ent_type 只能选：Person, Org, Law, Crime, Location, Time, Book, Concept, Other。\n"
        f"3) 每个句子最多输出 {MAX_CAND_PER_SENT} 个实体，按重要性降序。\n"
    )
    user = {"sentences": [{"sentence_id": x["sentence_id"], "text": x["text"]} for x in batch]}
    content = chat_with_retry(
        model=CANDIDATE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.1,
    )
    data = extract_json(content) or {}
    items = data.get("items", [])
    out: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sentence_id", "")).strip()
        ents = it.get("entities", [])
        if not sid or not isinstance(ents, list):
            continue
        res = []
        for e in ents:
            if isinstance(e, dict) and e.get("mention"):
                res.append({
                    "mention": str(e["mention"]).strip(),
                    "ent_type": str(e.get("ent_type", "Other")).strip() or "Other"
                })
        out[sid] = res[:MAX_CAND_PER_SENT]
    return out


def llm_verify_entities_batch(batch: List[Dict[str, Any]], cand_map: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    system = (
        "你是实体筛选器。请对多个句子分别从候选列表中筛选应保留的实体。\n"
        "输出必须是严格 JSON（不要 Markdown，不要解释），格式：\n"
        "{\"items\":[{\"sentence_id\":\"...\",\"entities\":[{\"mention\":\"...\",\"ent_type\":\"...\"}, ...]}, ...]}\n"
        "硬规则：\n"
        "1) mention 必须完全来自该句的候选实体列表（完全一致）。\n"
        "2) mention 必须是该句原文连续子串。\n"
        "3) 不要保留泛化名词（行为/规定/情况/方式/结果/因素/原则/制度/关系等）。\n"
        f"4) 每句最多保留 {MAX_TRUTH_PER_SENT} 个，按重要性降序。\n"
    )
    payload = {
        "items": [
            {"sentence_id": x["sentence_id"], "text": x["text"], "candidates": cand_map.get(x["sentence_id"], [])}
            for x in batch
        ]
    }
    content = chat_with_retry(
        model=VERIFY_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.1,
    )

    data = extract_json(content) or {}
    items = data.get("items", [])
    out: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("sentence_id", "")).strip()
        ents = it.get("entities", [])
        if not sid or not isinstance(ents, list):
            continue
        res = []
        for e in ents:
            if isinstance(e, dict) and e.get("mention"):
                res.append({
                    "mention": str(e["mention"]).strip(),
                    "ent_type": str(e.get("ent_type", "Other")).strip() or "Other"
                })
        out[sid] = res[:MAX_TRUTH_PER_SENT]
    return out


def main():
    print("========== Truth-Entity（batch + checkpoint + no examples）==========")
    sents = load_sentences(sent_tsv_path)
    print(f"📄 已加载句子数：{len(sents)}")
    print(f"🤖 候选模型：{CANDIDATE_MODEL} | 筛选模型：{VERIFY_MODEL}")
    print(f"⚙️ BATCH_SIZE={BATCH_SIZE}, MAX_CAND_PER_SENT={MAX_CAND_PER_SENT}, MAX_TRUTH_PER_SENT={MAX_TRUTH_PER_SENT}")
    if ENABLE_SKIP_VERIFY:
        print(f"⚡ cand<= {SKIP_VERIFY_IF_CAND_LEQ} 时跳过 R1")
    print(f"💾 checkpoint 输出：{truth_entity_tsv_path}\n")

    done_sids = load_done_sentence_ids(truth_entity_tsv_path)
    if done_sids:
        print(f"🔁 检测到已有输出文件，已完成 sentence 数：{len(done_sids)}（将自动跳过）")

    # 过滤掉已完成的句子
    todo = [x for x in sents if x["sentence_id"] not in done_sids]
    print(f"🧩 待处理句子数：{len(todo)}\n")
    if not todo:
        print("✅ 没有待处理内容，结束。")
        return

    total_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    global_eid = 1

    # 如果已有文件，为避免 entity_id 重复，简单做法：从已有行数推 eid
    if os.path.exists(truth_entity_tsv_path):
        with open(truth_entity_tsv_path, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1
        global_eid = max(1, n + 1)

    for bi in range(total_batches):
        batch = todo[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]

        # 1) 候选（V3）
        cand_items = llm_candidate_entities_batch(batch)

        # 2) 清洗候选（规则层）
        cand_mentions_map: Dict[str, List[str]] = {}
        for x in batch:
            sid = x["sentence_id"]
            text = x["text"]
            cand = cand_items.get(sid, [])

            cm = []
            for c in cand:
                m0 = clean_mention(c["mention"])
                for m in split_list(m0):
                    if is_bad(m):
                        continue
                    if find_span(text, m) is None:
                        continue
                    if m not in cm:
                        cm.append(m)
            cand_mentions_map[sid] = cm

        # 3) 筛选（R1）或跳过
        need_verify = []
        keep_map: Dict[str, List[Dict[str, str]]] = {}

        if ENABLE_SKIP_VERIFY:
            for x in batch:
                sid = x["sentence_id"]
                cm = cand_mentions_map.get(sid, [])
                if len(cm) <= SKIP_VERIFY_IF_CAND_LEQ:
                    keep_map[sid] = [{"mention": m, "ent_type": "Other"} for m in cm]
                else:
                    need_verify.append(x)
        else:
            need_verify = batch

        if need_verify:
            verify_in = {x["sentence_id"]: cand_mentions_map.get(x["sentence_id"], []) for x in need_verify}
            verified = llm_verify_entities_batch(need_verify, verify_in)
            for sid, ents in verified.items():
                keep_map[sid] = ents

        # 4) 生成 rows 并追加写入 checkpoint
        rows = []
        for x in batch:
            sid = x["sentence_id"]
            text = x["text"]
            page_no = x["page_no"]
            keep = keep_map.get(sid, [])

            for k in keep:
                m = clean_mention(k["mention"])
                sp = find_span(text, m)
                if sp is None:
                    continue
                ent_type = str(k.get("ent_type", "Other") or "Other")
                rows.append({
                    "entity_id": f"e{global_eid:05d}",
                    "sentence_id": sid,
                    "page_no": page_no,
                    "mention": m,
                    "start_char": sp[0],
                    "end_char": sp[1],
                    "ent_type": ent_type,
                    "confidence": DEFAULT_CONF_TRUTH,
                })
                global_eid += 1

        append_entities_tsv(rows, truth_entity_tsv_path)

        if (bi + 1) % PRINT_EVERY_N_BATCH == 0:
            print(f"…batch {bi+1}/{total_batches} 完成 | 本批写入 {len(rows)} 行 | 已输出文件累计更新")

    print("\n✅ 全部完成（已持续写入文件，不怕中断）。")


if __name__ == "__main__":
    main()
