import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ---- adjust these imports to match your project ----
# These should exist in your codebase:
# - Retriever: returns top-k docs in the same format you already use in app
# - AnswerGenerator: your class from rag/generate.py
#
# If your retriever class/function has a different name, change ONLY the import + call below.
from rag.index import VectorIndexer
from rag.retrieve import DocumentRetriever
from rag.generate import AnswerGenerator
# ----------------------------------------------------


@dataclass
class EvalResult:
    id: str
    question: str
    k: int
    has_answer: bool
    has_citations: bool
    citation_rate: float
    expected_contains_pass: bool
    expected_contains_missing: List[str]


def sentence_split(text: str) -> List[str]:
    # simple + transparent split (no NLP deps)
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


def citation_rate(answer: str) -> float:
    sents = sentence_split(answer)
    if not sents:
        return 0.0
    cited = 0
    for s in sents:
        if re.search(r"\[Source \d+\]", s):
            cited += 1
    return cited / len(sents)


def expected_contains_check(answer: str, expected_contains: List[str]) -> Tuple[bool, List[str]]:
    ans = (answer or "").lower()
    missing = [t for t in expected_contains if t.lower() not in ans]
    return (len(missing) == 0), missing


def load_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="eval/dataset.jsonl",
        help="Path to eval dataset (jsonl)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-K documents for retrieval"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Optional collection/index name"
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    # Instantiate components
    generator = AnswerGenerator()

    results: List[EvalResult] = []

    for row in dataset:
        qid = row.get("id", "unknown")
        question = row["question"]
        expected_contains = row.get("expected_contains", [])
        k = int(row.get("k", args.k))

        indexer = VectorIndexer(persist_directory="chroma_db")
        retriever = DocumentRetriever(indexer)
        retrieved_docs = retriever.retrieve(question, k=k)

        out = generator.generate_answer(question, retrieved_docs)

        answer = out.get("answer", "") or ""
        citations = out.get("citations", []) or []

        has_answer = bool(answer.strip()) and "Error generating answer:" not in answer
        has_cits = len(citations) > 0

        cr = citation_rate(answer)
        exp_pass, exp_missing = expected_contains_check(answer, expected_contains)

        results.append(
            EvalResult(
                id=qid,
                question=question,
                k=k,
                has_answer=has_answer,
                has_citations=has_cits,
                citation_rate=cr,
                expected_contains_pass=exp_pass,
                expected_contains_missing=exp_missing,
            )
        )

    # Summary
    total = len(results)
    if total == 0:
        print("No eval rows found.")
        return

    def pct(x: int) -> str:
        return f"{(x/total)*100:.1f}%"

    has_answer_cnt = sum(1 for r in results if r.has_answer)
    has_cits_cnt = sum(1 for r in results if r.has_citations)
    exp_pass_cnt = sum(1 for r in results if r.expected_contains_pass)
    avg_citation_rate = sum(r.citation_rate for r in results) / total

    print("\n=== RAG Eval Summary ===")
    print(f"Rows: {total}")
    print(f"Has answer: {has_answer_cnt}/{total} ({pct(has_answer_cnt)})")
    print(f"Has citations: {has_cits_cnt}/{total} ({pct(has_cits_cnt)})")
    print(f"Expected-contains pass (simple accuracy proxy): {exp_pass_cnt}/{total} ({pct(exp_pass_cnt)})")
    print(f"Avg citation rate (sentences with [Source N]): {avg_citation_rate:.2f}")

    print("\n=== Failures ===")
    any_fail = False
    for r in results:
        if not r.has_answer or not r.has_citations or not r.expected_contains_pass:
            any_fail = True
            print(f"- {r.id}:")
            if not r.has_answer:
                print("  - no answer / error")
            if not r.has_citations:
                print("  - no citations detected")
            if not r.expected_contains_pass:
                print(f"  - missing expected terms: {r.expected_contains_missing}")
    if not any_fail:
        print("None")

    print("")


if __name__ == "__main__":
    main()
