# /// script
# requires-python = ">=3.11"
# dependencies = ["FlagEmbedding", "numpy", "huggingface-hub"]
# ///
"""Local BGE-M3 retrieval. Download once, then index/retrieve/eval offline."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import local_rag


MODEL_DIR = local_rag.PROJECT_ROOT / "models/bge-m3"
# Recorded from the locally downloaded Hugging Face metadata (2026-11-07).
MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
INDEX_FILE = local_rag.PROJECT_ROOT / ".document-cache/bge_m3_index.npz"
_MODEL = None
_VECTORS = None


def model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_DIR.is_dir():
            raise SystemExit("找不到本地模型；請先執行 download，或用 --model-dir 指定模型資料夾。")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from FlagEmbedding import BGEM3FlagModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"本地模型：{MODEL_DIR.resolve()}；裝置：{device}", flush=True)
        _MODEL = BGEM3FlagModel(
            str(MODEL_DIR.resolve()), devices=device,
            use_fp16=device.startswith("cuda"),
        )
    return _MODEL


def encode(texts):
    import numpy as np

    encoder = model()
    # Refuse silent truncation, so comparisons use the same complete chunks.
    for text in texts:
        if len(encoder.tokenizer.encode(text, truncation=False)) > 8192:
            raise ValueError("輸入超過 8192 tokens；請調整切塊，不可直接截斷後比較。")
    vectors = np.asarray(encoder.encode(
        texts, batch_size=1, max_length=8192,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("模型產生無效向量。")
    return vectors / norms


def signature(records):
    # Include the exact chunks and model files; reject stale indexes.
    files = sorted(
        (str(p.relative_to(MODEL_DIR)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in MODEL_DIR.rglob("*") if p.is_file()
    )
    data = {"chunks": [r["text"] for r in records], "model": files,
            "path": str(MODEL_DIR.resolve()), "max_length": 8192, "version": 1}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def index():
    import numpy as np

    records, _, _ = local_rag.build_index()
    if not records:
        raise ValueError("文件沒有可索引的片段。")
    print(f"將 {len(records)} 個片段轉成向量（batch_size=1）。", flush=True)
    vectors = encode([r["text"] for r in records])
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = INDEX_FILE.with_suffix(".tmp.npz")
    np.savez_compressed(temp, vectors=vectors, signature=signature(records))
    temp.replace(INDEX_FILE)
    print(f"已保存：{INDEX_FILE}，shape={vectors.shape}")


def document_vectors(records):
    import numpy as np

    global _VECTORS
    if _VECTORS is None:
        if not INDEX_FILE.exists():
            raise SystemExit("尚未建立向量索引；請先執行 index。")
        with np.load(INDEX_FILE, allow_pickle=False) as saved:
            if str(saved["signature"]) != signature(records):
                raise SystemExit("文件或模型已改變；請重新執行 index。")
            _VECTORS = saved["vectors"]
        if _VECTORS.ndim != 2 or len(_VECTORS) != len(records):
            raise ValueError("索引與文件片段數量不符。")
    return _VECTORS


def fuse(rankings, pool=20, constant=60, weights=None):
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if pool < 1 or not math.isfinite(constant) or constant < 0:
        raise ValueError("候選數必須大於 0，RRF 常數必須是有限的非負數。")
    if (len(weights) != len(rankings) or not weights
            or any(not math.isfinite(w) or w < 0 for w in weights)
            or sum(weights) == 0):
        raise ValueError("每份排名需有一個有限非負權重，且至少一個權重大於 0。")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, passage in enumerate(ranking[:pool], 1):
            key = passage["start"]
            if key not in scores:
                scores[key] = {**passage, "score": 0.0}
            scores[key]["score"] += weight / (constant + rank)
    return sorted(scores.values(), key=lambda p: (-p["score"], p["start"]))


def retrieve(question, limit=5, mode="semantic"):
    if mode == "bm25":
        return local_rag.retrieve(question, limit)
    import numpy as np

    records, _, _ = local_rag.build_index()
    vectors = document_vectors(records)
    query = encode([question])[0]
    scores = vectors @ query
    order = np.argsort(-scores, kind="stable")
    semantic = [
        {**{k: v for k, v in records[i].items() if k != "counts"},
         "score": float(scores[i])}
        for i in order[:max(20, limit)]
    ]
    if mode == "hybrid":
        return fuse([local_rag.retrieve(question, max(20, limit)), semantic],
                    pool=max(20, limit))[:limit]
    return semantic[:limit]


def evaluate(testset, mode, output):
    ranks = {}
    for line in testset.read_text(encoding="utf-8-sig").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"Q\d{2}", cells[0]):
            continue
        match = re.search(r"快取第\s*(\d+)\s*[–-]\s*(\d+)\s*行", cells[3])
        if not match:
            continue
        start, end = map(int, match.groups())
        passages = retrieve(cells[1], 10, mode)
        rank = next((i for i, p in enumerate(passages, 1)
                     if p["start_line"] <= end and p["end_line"] >= start), None)
        ranks[cells[0]] = rank
        print(f"{cells[0]}: rank={rank or '-'}", flush=True)
    if not ranks:
        raise ValueError("測試集沒有包含標準來源行號的題目。")
    metrics = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks.values())
                                / len(ranks), 3) for k in (1, 3, 5, 10)}
    metrics["mrr@10"] = round(sum(1 / r for r in ranks.values() if r) / len(ranks), 3)
    result = {"mode": mode, "model": str(MODEL_DIR), "testset": str(testset),
              "document": str(local_rag.DOCUMENT), "questions": len(ranks),
              "metrics": metrics, "ranks": ranks,
              "hit_rule": "source line overlap", "rrf_pool": 20, "rrf_constant": 60}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"已保存：{output}")


def main():
    global MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="下載模型；僅此指令需要連線")
    commands.add_parser("index", help="將現有 Markdown 切塊並建立向量索引")
    sub = commands.add_parser("retrieve")
    sub.add_argument("question")
    sub.add_argument("-k", type=int, default=5)
    sub.add_argument("--mode", choices=("bm25", "semantic", "hybrid"), default="semantic")
    sub = commands.add_parser("eval")
    sub.add_argument("--mode", choices=("bm25", "semantic", "hybrid"), default="semantic")
    sub.add_argument("--testset", type=Path, default=local_rag.TESTSET)
    sub.add_argument("--output", type=Path)
    sub = commands.add_parser("rankings", help="保存每題兩種檢索的前 20 名與逐題 HTML 對照報告")
    sub.add_argument("--testset", type=Path, default=local_rag.TESTSET)
    sub.add_argument("--output-dir", type=Path, default=local_rag.PROJECT_ROOT / "outputs/rankings")
    sub = commands.add_parser("fusion-experiment", help="只讀保存的排名，比較常數與權重；不載入模型")
    sub.add_argument("--rankings", type=Path, required=True)
    sub.add_argument("--output-dir", type=Path, default=local_rag.PROJECT_ROOT / "outputs/experiments")
    sub = commands.add_parser("chunk-experiment", help="建立獨立切塊索引並評估三種檢索；融合固定60、1:1")
    sub.add_argument("--size", type=int, required=True, help="每塊字元數")
    sub.add_argument("--overlap", type=int, required=True, help="相鄰片段重疊字元數")
    sub.add_argument("--baseline", type=Path, help="選用：原設定的排名JSON，供同份報告比較")
    sub.add_argument("--testset", type=Path, default=local_rag.TESTSET)
    sub.add_argument("--output-dir", type=Path, default=local_rag.PROJECT_ROOT / "outputs/experiments/chunks")
    args = parser.parse_args()
    MODEL_DIR = args.model_dir
    if args.command == "download":
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL_DIR),
                          allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.pt",
                                          "1_Pooling/*"])
        print(f"已下載：{MODEL_DIR.resolve()}")
    elif args.command == "index":
        index()
    elif args.command == "eval":
        output = args.output or (local_rag.PROJECT_ROOT / "outputs/evaluations"
                                 / f"bge_eval_{args.mode}_{args.testset.stem}.json")
        evaluate(args.testset, args.mode, output)
    elif args.command == "rankings":
        from export_rankings import export_rankings
        export_rankings(args.testset, args.output_dir, sys.modules[__name__])
    elif args.command == "fusion-experiment":
        from fusion_experiment import run_experiment
        run_experiment(args.rankings, args.output_dir)
    elif args.command == "chunk-experiment":
        if args.size <= 0 or not 0 <= args.overlap < args.size:
            parser.error("--size 必須大於0，--overlap 必須非負且小於 --size")
        from chunk_experiment import run_chunk_experiment
        run_chunk_experiment(sys.modules[__name__], args.size, args.overlap,
                             args.testset, args.output_dir, args.baseline)
    else:
        if args.k <= 0:
            parser.error("-k 必須大於 0")
        for i, passage in enumerate(retrieve(args.question, args.k, args.mode), 1):
            print(f"\n[片段{i}] score={passage['score']:.6f} "
                  f"lines={passage['start_line']}-{passage['end_line']}")
            print(passage["text"])


if __name__ == "__main__":
    main()
