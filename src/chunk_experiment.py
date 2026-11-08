"""One chunking configuration per run; preserve the default index and settings."""

import hashlib
import html
import json
from datetime import datetime, timedelta, timezone

from export_rankings import export_rankings
from fusion_experiment import first_hit, metrics


def configure(engine, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk 大小必須大於0，且 0 <= overlap < size。")
    engine.local_rag.CHUNK_SIZE = size
    engine.local_rag.CHUNK_OVERLAP = overlap
    engine._VECTORS = None
    cache = engine.local_rag.PROJECT_ROOT / ".document-cache"
    engine.INDEX_FILE = (cache / "bge_m3_index.npz" if (size, overlap) == (3000, 500)
                         else cache / "chunks" / f"size_{size}_overlap_{overlap}.npz")


def summarize(snapshot, fuse):
    if snapshot.get("schema_version") != 1 or snapshot.get("pool_per_method") != 20:
        raise ValueError("比較用排名必須是 rankings 指令產生的前20名快照。")
    ranks = {mode: {} for mode in ("bm25", "semantic", "hybrid")}
    for q in snapshot["questions"]:
        if q["gold_lines"] is None:
            continue
        if q["id"] in ranks["bm25"]:
            raise ValueError(f"重複題號：{q['id']}")
        for mode in ("bm25", "semantic"):
            ps = q[mode]
            if len(ps) != 20 or [p["rank"] for p in ps] != list(range(1, 21)):
                raise ValueError(f"{q['id']} 的 {mode} 排名前20名不完整。")
            ranks[mode][q["id"]] = first_hit(ps, q["gold_lines"])
        fused = fuse([q["bm25"], q["semantic"]], pool=20, constant=60, weights=[1, 1])
        ranks["hybrid"][q["id"]] = first_hit(fused, q["gold_lines"])
    return {mode: {"metrics": metrics(values), "ranks": values} for mode, values in ranks.items()}


def validate_baseline(baseline, document_hash, testset_hash):
    for key, expected in (("document_sha256", document_hash), ("testset_sha256", testset_hash)):
        if baseline.get(key) != expected:
            raise ValueError("對照快照的文件或測試集與目前不同，不能直接比較。")


def render_report(result):
    esc = lambda x: html.escape(str(x), quote=True)
    rows, questions = [], []
    for group in result["groups"]:
        c = group["chunking"]
        for mode, run in group["runs"].items():
            m = run["metrics"]
            rows.append(f'<tr><td>{esc(group["label"])}</td><td>{c["size_chars"]}</td>'
                        f'<td>{c["overlap_chars"]}</td><td>{c["count"]}</td><td>{mode}</td>'
                        + ''.join(f'<td>{m[f"hit@{k}"]:.1%}</td>' for k in (1, 3, 5, 10))
                        + f'<td>{m["mrr@10"]:.3f}</td></tr>')
    qids = list(result["groups"][-1]["runs"]["bm25"]["ranks"])
    for qid in qids:
        questions.append(f'<tr><td>{qid}</td>' + ''.join(
            f'<td>{group["runs"][mode]["ranks"].get(qid) or "前10名外"}</td>'
            for group in result["groups"] for mode in ("bm25", "semantic", "hybrid")) + '</tr>')
    headings = ''.join(f'<th>{esc(group["label"])}<br>{mode}</th>'
                       for group in result["groups"] for mode in ("bm25", "semantic", "hybrid"))
    return f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Chunk 實驗</title>
    <style>body{{font:16px/1.6 "Microsoft JhengHei",sans-serif;background:#f6f8fa;color:#183042;margin:24px}}
    main{{max-width:1500px;margin:auto}}table{{border-collapse:collapse;background:white;width:100%;margin:20px 0}}
    th,td{{padding:10px;border:1px solid #ccd8df;text-align:left}}th{{background:#183d55;color:white}}
    .scroll{{overflow-x:auto}}.note{{background:#e5eef5;padding:16px}}p{{overflow-wrap:anywhere}}</style>
    <main><h1>第二階段：Chunk／重疊實驗</h1><p>產生時間：{esc(result['created_at'])}</p>
    <div class="note">單位為字元；不清理原文，維持同一模型與測試集。融合固定常數60、權重1:1、各前20名。<br>
    每組使用自己的切塊，同時比較 BM25、語義、混合；表中的所有名次與指標限制於前10名。<br>
    仍使用標準來源行號有交集的舊命中規則，不代表完整答案覆蓋；大小不同可能影響命中機會。<br>
    各組 C001 等編號只在各自切塊內有效，跨組請用原文行號對照。前20名包含的文字總量也隨切塊大小改變。<br>
    此為開發集比較，尚未套用第一階段調整後的融合設定。</div>
    <h2>指標比較</h2><div class="scroll"><table><thead><tr><th>組別</th><th>大小</th><th>重疊</th><th>片段數</th>
    <th>方法</th><th>Hit@1</th><th>Hit@3</th><th>Hit@5</th><th>Hit@10</th><th>MRR@10</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div><h2>逐題首次命中名次</h2>
    <div class="scroll"><table><thead><tr><th>題號</th>{headings}</tr></thead><tbody>{''.join(questions)}</tbody></table></div>
    <p>本次完整排行榜（含片段全文）：{esc(result['ranking_snapshot'])}<br>
    向量索引：{esc(result['index_file'])}</p></main></html>'''


def run_chunk_experiment(engine, size, overlap, testset, output_dir, baseline_file=None):
    baseline = None
    document_hash = hashlib.sha256(engine.local_rag.DOCUMENT.read_bytes()).hexdigest()
    testset_hash = hashlib.sha256(testset.read_bytes()).hexdigest()
    if baseline_file is not None:
        baseline = json.loads(baseline_file.read_bytes())
        validate_baseline(baseline, document_hash, testset_hash)
        baseline_runs = summarize(baseline, engine.fuse)  # Validate before any expensive work.
    configure(engine, size, overlap)
    records, _, _ = engine.local_rag.build_index()
    print(f"Chunk={size}；重疊={overlap}；共 {len(records)} 個片段。", flush=True)
    if engine.INDEX_FILE.exists():
        engine.document_vectors(records)
        print(f"沿用已驗證的索引：{engine.INDEX_FILE}", flush=True)
    else:
        engine.index()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"size_{size}_overlap_{overlap}" / stamp
    ranking_file = export_rankings(testset, run_dir, engine)
    snapshot = json.loads(ranking_file.read_bytes())
    groups = []
    if baseline is not None:
        groups.append({"label": "原始對照", "chunking": baseline["chunking"],
                       "ranking_snapshot": str(baseline_file.resolve()),
                       "snapshot_sha256": hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
                       "runs": baseline_runs})
    groups.append({"label": "本次切塊", "chunking": snapshot["chunking"],
                   "ranking_snapshot": str(ranking_file.resolve()),
                   "snapshot_sha256": hashlib.sha256(ranking_file.read_bytes()).hexdigest(),
                   "runs": summarize(snapshot, engine.fuse)})
    result = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "groups": groups, "index_file": str(engine.INDEX_FILE.resolve()),
              "ranking_snapshot": str(ranking_file.resolve()),
              "model_id": engine.MODEL_ID, "model_revision": engine.MODEL_REVISION,
              "document_sha256": document_hash, "testset_sha256": testset_hash,
              "rrf_constant": 60, "rrf_weights": [1, 1], "pool_per_method": 20,
              "cutoff": 10, "hit_rule": "source line overlap"}
    summary_file = run_dir / "summary.json"
    summary_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_file.with_suffix(".html").write_text(render_report(result), encoding="utf-8")
    for group in groups:
        for mode, run in group["runs"].items():
            print(f"{group['label']} {mode}: {json.dumps(run['metrics'])}")
    print(f"比較報告：{summary_file.with_suffix('.html').resolve()}")
    print(f"完整排名：{ranking_file.resolve()}")
