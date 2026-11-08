"""Offline checks: python -m unittest discover -s tests. No model download needed."""

import contextlib
import io
import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bge_local
import export_rankings
import fusion_experiment


class RetrievalChecks(unittest.TestCase):
    def test_weighted_rrf_math_and_invalid_settings(self):
        a, b = [{"start": i, "score": 100} for i in range(2)]
        result = bge_local.fuse([[a, b], [b, a]], constant=10, weights=[2, 1])
        self.assertEqual(result[0]["start"], 0)
        self.assertAlmostEqual(result[0]["score"], 2 / 11 + 1 / 12)
        self.assertEqual(a["score"], 100)
        only_bm25 = bge_local.fuse([[a, b], [b, a]], weights=[1, 0])
        self.assertEqual([p["start"] for p in only_bm25], [0, 1])
        for settings in ({"constant": -1}, {"constant": float("nan")},
                         {"weights": [0, 0]}, {"weights": [1]}, {"weights": [1, -1]}):
            with self.assertRaises(ValueError):
                bge_local.fuse([[a], [b]], **settings)

    def test_frozen_fusion_experiment_uses_top10_and_skips_unscored(self):
        passages = [{"start": i * 10, "start_line": i + 1, "end_line": i + 1,
                     "chunk_id": f"C{i+1:03d}", "score": 1.0, "text": "原文"}
                    for i in range(21)]
        annotate = lambda ps: [{**p, "rank": i} for i, p in enumerate(ps, 1)]
        question = {"id": "Q01", "question": "問題", "gold_lines": [1, 1],
                    "bm25": annotate(passages[:20]), "semantic": annotate(passages[1:])}
        snapshot = {"schema_version": 1, "pool_per_method": 20,
                    "questions": [question, {"id": "Q30", "gold_lines": None}]}
        result = fusion_experiment.compare(snapshot)
        self.assertEqual(result["scored_questions"], 1)
        self.assertEqual(result["unscored_questions"], ["Q30"])
        self.assertEqual(result["runs"]["bm25"]["ranks"], {"Q01": 1})
        self.assertIsNone(result["runs"]["semantic"]["ranks"]["Q01"])
        # Shared candidates outrank the BM25-only #1 at the original c=60.
        self.assertIsNone(result["runs"]["original"]["ranks"]["Q01"])
        self.assertEqual(result["runs"]["original"]["metrics"]["mrr@10"], 0)
        original = result["questions"][0]["trials"]["original"]
        self.assertEqual([p["start"] for p in original],
                         [p["start"] for p in bge_local.fuse([question["bm25"], question["semantic"]])])
        for trial in fusion_experiment.TRIALS:
            for p in result["questions"][0]["trials"][trial["id"]]:
                self.assertAlmostEqual(p["rrf_score"], p["bm25_rrf"] + p["semantic_rrf"])
        result.update(snapshot_file="test.json", created_at="test", chunking={"size_chars": 3000})
        self.assertIn('第一階段', fusion_experiment.render_report(result))

    def test_rankings_snapshot_retains_both_lists_and_unscored_questions(self):
        questions = export_rankings.questions_from(bge_local.local_rag.TESTSET)
        self.assertEqual(len(questions), 30)
        self.assertEqual(sum(q["gold_lines"] is not None for q in questions), 24)
        a = {"start": 0, "end": 20, "start_line": 1, "end_line": 10,
             "text": '<script>alert("raw text")</script>', "score": 0.5}
        b = {"start": 20, "end": 40, "start_line": 11, "end_line": 20,
             "text": "另一段原文", "score": 0.2}
        test_questions = [
            {**questions[0], "question": "<unsafe>", "gold_lines": [10, 10]},
            questions[-1],
        ]
        calls = []
        def fake_retrieve(question, limit, mode):
            calls.append((question, limit, mode))
            return [a, b] if mode == "bm25" else [b, a]
        with contextlib.redirect_stdout(io.StringIO()):
            result = export_rankings.collect(test_questions, fake_retrieve, {0: "C001", 20: "C002"})
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(limit == 20 for _, limit, _ in calls))
        self.assertEqual(result[0]["first_source_overlap_rank"],
                         {"bm25": 1, "semantic": 2, "hybrid": 1})
        self.assertTrue(result[0]["bm25"][0]["source_overlap"])
        self.assertFalse(result[0]["bm25"][1]["source_overlap"])
        self.assertIsNone(result[1]["first_source_overlap_rank"])
        self.assertIsNone(result[1]["semantic"][0]["source_overlap"])
        self.assertEqual([p["start"] for p in result[0]["rrf_comparison"]],
                         [p["start"] for p in bge_local.fuse([[a, b], [b, a]])])
        report = export_rankings.render_report({"questions": result,
                                               "created_at": "test", "chunking": {"count": 2}})
        self.assertIn('&lt;script&gt;', report)
        self.assertNotIn('<script>alert(', report)
        self.assertIn('不評定', report)
        self.assertIn('另一段原文', report)

    def test_rrf_rewards_agreement_and_deduplicates(self):
        a, b, c = [{"start": i, "text": str(i), "score": 100} for i in range(3)]
        result = bge_local.fuse([[a, b], [c, b]])
        self.assertEqual([p["start"] for p in result], [1, 0, 2])
        self.assertAlmostEqual(result[0]["score"], 2 / 62)
        self.assertEqual(a["score"], 100)  # Do not mutate BM25 scores.

    def test_bm25_evaluation_matches_saved_baseline(self):
        # Expected regression values are test assertions, not an external output dependency.
        baseline_file = bge_local.local_rag.EVAL_FILE
        original = baseline_file.read_bytes() if baseline_file.exists() else None
        expected_ranks = {f"Q{i:02d}": 1 for i in range(1, 25)}
        expected_ranks.update(Q01=9, Q07=4, Q11=3, Q21=2)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            output = Path(folder) / "eval.json"
            with contextlib.redirect_stdout(io.StringIO()):
                bge_local.evaluate(bge_local.local_rag.TESTSET, "bm25", output)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["questions"], 24)
        self.assertEqual(result["ranks"], expected_ranks)
        self.assertEqual(result["metrics"], {
            "hit@1": 0.833, "hit@3": 0.917, "hit@5": 0.958,
            "hit@10": 1.0, "mrr@10": 0.883,
        })
        if original is not None:
            self.assertEqual(baseline_file.read_bytes(), original)
        else:
            self.assertFalse(baseline_file.exists())

    def test_empty_evaluation_does_not_write_result(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            testset = Path(folder) / "empty.md"
            testset.write_text("no questions", encoding="utf-8")
            output = Path(folder) / "eval.json"
            with self.assertRaises(ValueError):
                bge_local.evaluate(testset, "bm25", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
