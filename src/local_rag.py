# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
import bisect
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = PROJECT_ROOT / (
    "data/processed/10005臺南市雨水下水道檢討規劃報告-38a259a6ddfa.md"
)
EVAL_FILE = PROJECT_ROOT / "outputs/evaluations/retrieval_eval.json"
TESTSET = PROJECT_ROOT / "data/testsets/rag_testset_v1.md"
_INDEX = None


def terms(text: str) -> list[str]:
    compact = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%]+", text.lower()))
    return [compact[index:index + 2] for index in range(len(compact) - 1)]


def chunks(text: str, size: int = 3000, overlap: int = 500):
    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size]
        if chunk.strip():
            yield start, chunk


def build_index():
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    text = DOCUMENT.read_text(encoding="utf-8")
    line_starts = [0] + [index + 1 for index, char in enumerate(text) if char == "\n"]
    records = [
        {
            "start": start,
            "end": start + len(chunk),
            "start_line": bisect.bisect_right(line_starts, start),
            "end_line": bisect.bisect_right(line_starts, start + len(chunk) - 1),
            "text": chunk,
            "counts": Counter(terms(chunk)),
        }
        for start, chunk in chunks(text)
    ]
    document_frequency = Counter(
        token for record in records for token in record["counts"]
    )
    average_length = sum(sum(record["counts"].values()) for record in records) / len(records)
    _INDEX = records, document_frequency, average_length
    return _INDEX


def retrieve(question: str, limit: int = 5) -> list[dict]:
    records, document_frequency, average_length = build_index()
    query = Counter(terms(question))
    scored = []

    for record in records:
        length = sum(record["counts"].values())
        score = 0.0
        for token, query_frequency in query.items():
            frequency = record["counts"].get(token, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(records) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += query_frequency * inverse_frequency * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * length / average_length)
            )
        scored.append((score, record))

    best = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return [
        {
            "score": round(score, 3),
            "start": record["start"],
            "end": record["end"],
            "start_line": record["start_line"],
            "end_line": record["end_line"],
            "text": record["text"],
        }
        for score, record in best
    ]


def evaluate() -> None:
    questions = []
    for line in TESTSET.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or not re.fullmatch(r"Q\d{2}", cells[0]):
            continue
        match = re.search(r"快取第\s*(\d+)\s*[–-]\s*(\d+)\s*行", cells[3])
        if match:
            questions.append((cells[0], cells[1], int(match[1]), int(match[2])))

    ranks = {}
    for question_id, question, gold_start, gold_end in questions:
        passages = retrieve(question, 10)
        rank = next(
            (
                index
                for index, passage in enumerate(passages, 1)
                if passage["start_line"] <= gold_end
                and passage["end_line"] >= gold_start
            ),
            None,
        )
        ranks[question_id] = rank
        print(f"{question_id}: rank={rank or '-'}")

    metrics = {
        f"hit@{limit}": round(
            sum(rank is not None and rank <= limit for rank in ranks.values())
            / len(ranks),
            3,
        )
        for limit in (1, 3, 5, 10)
    }
    metrics["mrr@10"] = round(
        sum(1 / rank for rank in ranks.values() if rank is not None) / len(ranks),
        3,
    )
    result = {"questions": len(ranks), "metrics": metrics, "ranks": ranks}
    EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    subparser = commands.add_parser("retrieve")
    subparser.add_argument("question", nargs="+")
    subparser.add_argument("-k", "--top-k", type=int, default=5)
    commands.add_parser("eval")
    args = parser.parse_args()

    if args.command == "eval":
        evaluate()
        return

    question = " ".join(args.question)
    if args.command == "retrieve":
        for index, passage in enumerate(retrieve(question, args.top_k), 1):
            preview = passage["text"].replace("\n", " ")[:300]
            print(
                f"[片段{index}] score={passage['score']} "
                f"lines={passage['start_line']}-{passage['end_line']} {preview}\n"
            )


if __name__ == "__main__":
    main()
