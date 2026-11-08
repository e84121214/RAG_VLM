"""Stage 1: controlled RRF experiments over a frozen top-20 ranking snapshot."""

import hashlib
import html
import json
from datetime import datetime, timedelta, timezone

from bge_local import fuse


# Each trial changes one factor relative to the original configuration.
TRIALS = [
    {"id": "hybrid_original", "label": "原設定", "constant": 60, "weights": [1, 1]},
    {"id": "hybrid_constant_10", "label": "A：只改常數", "constant": 10, "weights": [1, 1]},
    {"id": "hybrid_bm25_weight_2", "label": "B：只改權重", "constant": 60, "weights": [2, 1]},
    {"id": "hybrid_constant_10_hybrid_bm25_weight_2", "label": "C：同時改常數跟權重", "constant": 10, "weights": [2, 1]},
]


def first_hit(passages, gold):
    return next((i for i, p in enumerate(passages[:10], 1)
                 if p["start_line"] <= gold[1] and p["end_line"] >= gold[0]), None)


def metrics(ranks):
    if not ranks:
        raise ValueError("排名紀錄沒有可評估的標準來源題目。")
    result = {f"hit@{k}": round(sum(r is not None and r <= k for r in ranks.values())
                               / len(ranks), 3) for k in (1, 3, 5, 10)}
    result["mrr@10"] = round(sum(1 / r for r in ranks.values() if r) / len(ranks), 3)
    return result


def compare(snapshot):
    if snapshot.get("schema_version") != 1 or snapshot.get("pool_per_method") != 20:
        raise ValueError("需要 rankings 指令產生的 schema_version=1、各前20名快照。")
    ranks = {key: {} for key in ["bm25", "semantic"] + [t["id"] for t in TRIALS]}
    details = []
    skipped = []
    seen = set()
    for q in snapshot["questions"]:
        if q["id"] in seen:
            raise ValueError(f"重複題號：{q['id']}")
        seen.add(q["id"])
        gold = q["gold_lines"]
        if gold is None:
            skipped.append(q["id"])
            continue
        for method in ("bm25", "semantic"):
            passages = q[method]
            if len(passages) != 20 or len({p["start"] for p in passages}) != 20:
                raise ValueError(f"{q['id']} 的 {method} 必須保存20個不重複候選。")
            if [p["rank"] for p in passages] != list(range(1, 21)):
                raise ValueError(f"{q['id']} 的 {method} 排名順序不完整。")
            ranks[method][q["id"]] = first_hit(passages, gold)
        original_ranks = {method: {p["start"]: p["rank"] for p in q[method]}
                          for method in ("bm25", "semantic")}
        question = {"id": q["id"], "question": q["question"], "gold_lines": gold, "trials": {}}
        for trial in TRIALS:
            fused = fuse([q["bm25"], q["semantic"]], pool=20,
                         constant=trial["constant"], weights=trial["weights"])
            ranks[trial["id"]][q["id"]] = first_hit(fused, gold)
            candidates = []
            for rank, p in enumerate(fused, 1):
                br = original_ranks["bm25"].get(p["start"])
                sr = original_ranks["semantic"].get(p["start"])
                candidates.append({
                    "rank": rank, "chunk_id": p["chunk_id"],
                    "start": p["start"], "start_line": p["start_line"], "end_line": p["end_line"],
                    "bm25_rank": br, "semantic_rank": sr,
                    "bm25_rrf": trial["weights"][0] / (trial["constant"] + br) if br else 0.0,
                    "semantic_rrf": trial["weights"][1] / (trial["constant"] + sr) if sr else 0.0,
                    "rrf_score": p["score"],
                    "source_overlap": p["start_line"] <= gold[1] and p["end_line"] >= gold[0],
                })
            question["trials"][trial["id"]] = candidates
        details.append(question)
    runs = {}
    baseline = ranks["hybrid_original"]
    for key, values in ranks.items():
        deltas = {"improved": [], "worsened": [], "unchanged": []}
        for qid, rank in values.items():
            before, after = baseline[qid] or 11, rank or 11
            label = "improved" if after < before else "worsened" if after > before else "unchanged"
            deltas[label].append(qid)
        runs[key] = {"metrics": metrics(values), "ranks": values,
                     "versus_original": deltas}
    return {"trials": TRIALS, "runs": runs, "questions": details,
            "scored_questions": len(details), "unscored_questions": skipped}


def render_report(result):
    esc = lambda x: html.escape(str(x), quote=True)
    labels = {"bm25": "BM25 對照", "semantic": "純語義對照",
              **{t["id"]: t["label"] for t in TRIALS}}
    trial_by_id = {t["id"]: t for t in TRIALS}
    summary = []
    for key, run in result["runs"].items():
        trial = trial_by_id.get(key)
        settings = f"c={trial['constant']}；BM25:語義={trial['weights'][0]}:{trial['weights'][1]}" if trial else '單一方法'
        m = run["metrics"]
        summary.append(f'<tr><td>{esc(labels[key])}</td><td>{settings}</td>'
                       + ''.join(f'<td>{m[f"hit@{k}"]:.1%}</td>' for k in (1, 3, 5, 10))
                       + f'<td>{m["mrr@10"]:.3f}</td></tr>')
    rows = []
    sections = []
    for q in result["questions"]:
        rows.append(f'<tr><td><a href="#{q["id"]}">{q["id"]}</a></td><td>{esc(q["question"])}</td>'
                    + ''.join(f'<td>{run["ranks"][q["id"]] or "前10名外"}</td>'
                              for run in result["runs"].values()) + '</tr>')
        tables = []
        for trial in TRIALS:
            lines = []
            for p in q["trials"][trial["id"]][:10]:
                lines.append(f'<tr class="{"hit" if p["source_overlap"] else ""}"><td>{p["rank"]}</td>'
                             f'<td>{esc(p["chunk_id"])}（{p["start_line"]}–{p["end_line"]}行）</td>'
                             f'<td>{p["bm25_rank"] or "未入前20"}</td><td>{p["semantic_rank"] or "未入前20"}</td>'
                             f'<td>{p["bm25_rrf"]:.6f}</td><td>{p["semantic_rrf"]:.6f}</td>'
                             f'<td>{p["rrf_score"]:.6f}</td><td>{"來源行號命中" if p["source_overlap"] else "—"}</td></tr>')
            tables.append(f'<h3>{esc(trial["label"])}</h3><table><thead><tr><th>名次</th><th>片段</th>'
                          '<th>BM25名次</th><th>語義名次</th><th>BM25加分</th><th>語義加分</th><th>合計</th><th>命中</th>'
                          f'</tr></thead><tbody>{"".join(lines)}</tbody></table>')
        sections.append(f'<details id="{q["id"]}"><summary>{q["id"]}｜{esc(q["question"])}</summary>'
                        + ''.join(tables) + '</details>')
    return f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>第一階段：RRF 融合實驗</title>
    <style>body{{font:16px/1.6 "Microsoft JhengHei",sans-serif;background:#f6f8fa;color:#183042;margin:24px}}
    main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;background:white;margin:16px 0}}
    th,td{{padding:9px;border:1px solid #ccd8df;text-align:left;vertical-align:top}}th{{background:#183d55;color:white}}
    details{{padding:16px;background:white;margin:16px 0;border:1px solid #ccd8df}}summary{{cursor:pointer;font-weight:bold}}
    .hit{{background:#e1f3e9}}.note{{padding:16px;background:#e6eef5}}a{{color:#075f9b}}.scroll{{overflow-x:auto}}
    </style><main><h1>第一階段：RRF 常數與權重實驗</h1>
    <p>來源排名：{esc(result['snapshot_file'])}<br>產生時間：{esc(result['created_at'])}</p>
    <div class="note">固定使用同一份各前20名的排名快照；未重新推論、切塊或修改文件。<br>
    切塊設定：{esc(result['chunking'])}<br>
    評估 {result['scored_questions']} 題；略過 {len(result['unscored_questions'])} 題無標準來源題。<br>
    使用原有「來源行號交集」規則；所有名次與 MRR 均限制在前10名。<br>
    此為開發集的參數比較，並非新題目驗證；最終設定尚未選定或套用至日常檢索。</div>
    <h2>整體比較</h2><div class="scroll"><table><thead><tr><th>組別</th><th>設定</th><th>Hit@1</th><th>Hit@3</th>
    <th>Hit@5</th><th>Hit@10</th><th>MRR@10</th></tr></thead><tbody>{''.join(summary)}</tbody></table></div>
    <h2>逐題比較</h2><div class="scroll"><table><thead><tr><th>題號</th><th>問題</th>
    {''.join(f'<th>{esc(label)}</th>' for label in labels.values())}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    <h2>各題加分明細</h2><p>點題號展開；完整候選排名保存在同名 JSON，片段全文在原始排行榜。</p>{''.join(sections)}
    <script>document.querySelectorAll('a[href^="#Q"]').forEach(a=>a.addEventListener('click',()=>{{document.getElementById(a.hash.slice(1)).open=true;}}));</script>
    </main></html>'''


def run_experiment(snapshot_file, output_dir):
    raw = snapshot_file.read_bytes()
    snapshot = json.loads(raw)
    result = compare(snapshot)
    result.update(
        created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        snapshot_file=str(snapshot_file.resolve()), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        chunking=snapshot["chunking"], document_sha256=snapshot.get("document_sha256"),
        testset_sha256=snapshot.get("testset_sha256"), index_signature=snapshot.get("index_signature"),
        pool_per_method=20, hit_rule="source line overlap", cutoff=10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_dir / f"fusion_{stamp}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".html").write_text(render_report(result), encoding="utf-8")
    for key, run in result["runs"].items():
        print(f"{key}: {json.dumps(run['metrics'], ensure_ascii=False)}")
    print(f"已保存 JSON：{output.resolve()}")
    print(f"瀏覽器報告：{output.with_suffix('.html').resolve()}")
