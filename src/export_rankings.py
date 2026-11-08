"""Readable, offline snapshots of original rankings; no retrieval settings changed."""

import hashlib
import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def questions_from(testset):
    questions = []
    seen = set()
    for line in testset.read_text(encoding="utf-8-sig").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not re.fullmatch(r"Q\d{2}", cells[0]):
            continue
        if cells[0] in seen:
            raise ValueError(f"測試集題號重複：{cells[0]}")
        seen.add(cells[0])
        match = re.search(r"快取第\s*(\d+)\s*[–-]\s*(\d+)\s*行", cells[3])
        gold = list(map(int, match.groups())) if match else None
        if gold and not 1 <= gold[0] <= gold[1]:
            raise ValueError(f"標準來源行號不合法：{cells[0]}")
        questions.append({"id": cells[0], "question": cells[1],
                          "reference": cells[2], "source": cells[3], "gold_lines": gold})
    if not questions:
        raise ValueError("測試集沒有題目。")
    return questions


def annotate(passages, gold_lines, chunk_ids):
    return [
        {**p, "chunk_id": chunk_ids[p["start"]], "rank": rank,
         "source_overlap": (p["start_line"] <= gold_lines[1]
                            and p["end_line"] >= gold_lines[0]) if gold_lines else None}
        for rank, p in enumerate(passages, 1)
    ]


def comparison(bm25, semantic, gold_lines):
    candidates = {}
    for method, passages in (("bm25", bm25), ("semantic", semantic)):
        for p in passages:
            key = p["chunk_id"]
            if key not in candidates:
                candidates[key] = {k: p[k] for k in (
                    "chunk_id", "start", "end", "start_line", "end_line", "source_overlap")}
                candidates[key].update(bm25_rank=None, semantic_rank=None,
                                       bm25_rrf=0.0, semantic_rrf=0.0)
            candidates[key][f"{method}_rank"] = p["rank"]
            candidates[key][f"{method}_rrf"] = 1 / (60 + p["rank"])
    ranked = sorted(candidates.values(), key=lambda p: (
        -(p["bm25_rrf"] + p["semantic_rrf"]), p["start"]))
    for rank, p in enumerate(ranked, 1):
        p.update(hybrid_rank=rank, rrf_score=p["bm25_rrf"] + p["semantic_rrf"])
    first = lambda rows, rank_key: next(
        (p[rank_key] for p in rows if p["source_overlap"]), None)
    summary = None if gold_lines is None else {
        "bm25": first(bm25, "rank"), "semantic": first(semantic, "rank"),
        "hybrid": first(ranked, "hybrid_rank"),
    }
    return ranked, summary


def collect(questions, retriever, chunk_ids):
    results = []
    for question in questions:
        print(f"{question['id']}：取得 BM25 與語義檢索前 20 名…", flush=True)
        bm25 = annotate(retriever(question["question"], 20, "bm25"),
                        question["gold_lines"], chunk_ids)
        semantic = annotate(retriever(question["question"], 20, "semantic"),
                            question["gold_lines"], chunk_ids)
        fused, summary = comparison(bm25, semantic, question["gold_lines"])
        results.append({**question, "bm25": bm25, "semantic": semantic,
                        "rrf_comparison": fused, "first_source_overlap_rank": summary})
    return results


def render_report(data):
    esc = lambda value: html.escape(str(value), quote=True)

    def badge(overlap):
        if overlap is None:
            return '<span class="muted">不評定</span>'
        return '<span class="hit">來源行號命中</span>' if overlap else '未命中標準行號'

    def rank_label(rank):
        return str(rank) if rank is not None else '前 20 名外'

    def passages_table(title, passages):
        rows = []
        for p in passages:
            rows.append(f'''<tr><td>{p['rank']}</td><td>{esc(p['chunk_id'])}<br>
                第 {p['start_line']}–{p['end_line']} 行<br>{badge(p['source_overlap'])}</td>
                <td>{p['score']:.6f}</td><td><details><summary>{esc(p['text'][:90].replace(chr(10), ' '))}…（展開全文）</summary>
                <pre>{esc(p['text'])}</pre></details></td></tr>''')
        return f'<h3>{title}</h3><table><thead><tr><th>名次</th><th>片段／來源</th><th>原始分數</th><th>完整原文（可展開）</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'

    overview = []
    sections = []
    for q in data["questions"]:
        summary = q["first_source_overlap_rank"]
        values = [rank_label(summary[m]) if summary else '不評定'
                  for m in ("bm25", "semantic", "hybrid")]
        if summary and summary["hybrid"] is None:
            values[2] = '候選中無命中'
        overview.append(f'<tr><td><a href="#{q["id"]}">{q["id"]}</a></td><td>{esc(q["question"])}</td>'
                        + ''.join(f'<td>{value}</td>' for value in values) + '</tr>')
        fusion_rows = ''.join(
            f'<tr><td>{p["hybrid_rank"]}</td><td>{esc(p["chunk_id"])}（{p["start_line"]}–{p["end_line"]} 行）</td>'
            f'<td>{p["bm25_rank"] or "未入前20"}</td><td>{p["semantic_rank"] or "未入前20"}</td>'
            f'<td>{p["bm25_rrf"]:.6f}</td><td>{p["semantic_rrf"]:.6f}</td>'
            f'<td>{p["rrf_score"]:.6f}</td><td>{badge(p["source_overlap"])}</td></tr>'
            for p in q["rrf_comparison"])
        sections.append(f'''<details class="question" id="{q['id']}" {'open' if q['id'] == 'Q08' else ''}>
            <summary>{q['id']}｜{esc(q['question'])}</summary>
            <p><b>標準答案／預期行為：</b>{esc(q['reference'])}<br>
            <b>標準來源／所需資料：</b>{esc(q['source'])}</p>
            <div class="columns">{passages_table('BM25 前 20 名', q['bm25'])}
            {passages_table('BGE-M3 語義前 20 名', q['semantic'])}</div>
            <details><summary>查看原設定 RRF 加分明細（候選聯集）</summary>
            <div class="scroll"><table><thead><tr><th>混合名次</th><th>片段</th><th>BM25名次</th><th>語義名次</th>
            <th>BM25加分</th><th>語義加分</th><th>合計</th><th>來源行號</th></tr></thead>
            <tbody>{fusion_rows}</tbody></table></div></details></details>''')
    gold_count = sum(q["gold_lines"] is not None for q in data["questions"])
    return f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>逐題來源片段排行榜</title><style>
    body{{font:16px/1.6 "Microsoft JhengHei",sans-serif;margin:24px;color:#183042;background:#f6f8fa}}
    main{{max-width:1800px;margin:auto}}h1{{font-size:26px}}h3{{margin:12px 0}}
    table{{border-collapse:collapse;width:100%;background:white;margin:12px 0 24px;font-size:14px}}
    th,td{{padding:10px;border:1px solid #d5dfe7;text-align:left;vertical-align:top}}
    th{{background:#183d55;color:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.7;font-family:inherit}}
    .hit{{color:#11623e;font-weight:bold}}.muted{{color:#61717c}}summary{{cursor:pointer}}
    .question{{background:white;padding:16px;border:1px solid #d5dfe7;border-radius:8px;margin:16px 0}}
    .question>summary{{font-weight:bold;font-size:18px}}.scroll{{overflow-x:auto}}
    td:nth-child(2){{min-width:135px}}td:last-child{{overflow-wrap:anywhere}}
    a{{color:#075f9b}}.note{{background:#e8f0f6;padding:16px;border-radius:8px}}
    </style><main><h1>逐題來源片段排行榜</h1>
    <p>產生時間：{esc(data['created_at'])}｜模型：BGE-M3｜片段數：{data['chunking']['count']}</p>
    <div class="note">每題保存兩種方法各前 20 名。片段編號對應同一份文本切塊，可跨方法比對。<br>
    每塊最多 3,000 字元，重疊 500 字元，步長 2,500 字元。按原文包含換行的字元位置切分。<br>
    「來源行號命中」僅表示與標準行號有交集，不代表答案完整、語義正確或可拒答。<br>
    {gold_count} 題有標準來源；{len(data['questions']) - gold_count} 題無標準來源，不評定命中。原始 BM25 與語義分數不能直接比較。<br>
    RRF 維持各前 20 名、常數 60、等權；沒有進某方法前 20 名，該方法加分為 0。<br>
    總覽的 BM25／語義未命中指前 20 名外；混合未命中指候選聯集無標準來源（不是舊指標的前十名門檻）。</div>
    <h2>逐題首次命中名次</h2><p>點題號可展開該題。展開片段可讀完整原文；Q08 預先展開。</p>
    <table><thead><tr><th>題號</th><th>問題</th><th>BM25</th><th>語義</th><th>原設定混合</th></tr></thead>
    <tbody>{''.join(overview)}</tbody></table>{''.join(sections)}
    <script>document.querySelectorAll('a[href^="#Q"]').forEach(a=>a.addEventListener('click',()=>{{document.getElementById(a.hash.slice(1)).open=true;}}));</script>
    </main></html>'''


def export_rankings(testset, output_dir, engine):
    questions = questions_from(testset)
    records, _, _ = engine.local_rag.build_index()
    chunk_ids = {r["start"]: f"C{i:03d}" for i, r in enumerate(records, 1)}
    results = collect(questions, engine.retrieve, chunk_ids)
    data = {
        "schema_version": 1,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "testset": str(testset), "testset_sha256": hashlib.sha256(testset.read_bytes()).hexdigest(),
        "document": str(engine.local_rag.DOCUMENT),
        "document_sha256": hashlib.sha256(engine.local_rag.DOCUMENT.read_bytes()).hexdigest(),
        "model_dir": str(engine.MODEL_DIR.resolve()),
        "index_signature": engine.signature(records),
        "chunking": {"size_chars": 3000, "overlap_chars": 500, "stride_chars": 2500, "count": len(records)},
        "pool_per_method": 20, "rrf_constant": 60, "rrf_weights": [1, 1],
        "hit_rule": "source line overlap", "questions": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"rankings_{stamp}.json"
    html_path = json_path.with_suffix(".html")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_report(data), encoding="utf-8")
    print(f"已保存原始排名：{json_path.resolve()}")
    print(f"請用瀏覽器開啟對照報告：{html_path.resolve()}")
