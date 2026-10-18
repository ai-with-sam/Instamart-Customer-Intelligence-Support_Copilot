
from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

# Allow importing mock_llm.py from the project root.
sys.path.insert(0, str(ROOT))
import mock_llm


SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.
Answer the user's question using only the supplied policy context.
Do not use outside knowledge or invent policy details.
If the context does not contain the answer, reply exactly:
I could not find that information in the policy documents.
Return only the answer, without extra commentary.
"""


def load_chunks(overlap=10):
    """Split each policy into 40-word chunks with the requested overlap."""
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()

        if not words:
            continue

        step = 40 - overlap
        start = 0

        while True:
            chunk_words = words[start:start + 40]
            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words),
            })

            # Stop after the first chunk reaching the end of the document.
            if start + 40 >= len(words):
                break

            start += step

    return chunks


class PolicyRetriever:
    def __init__(self, overlap=10):
        self.chunks = load_chunks(overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        # Stable sorting keeps results reproducible when scores tie.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i)
        )[:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"],
            }
            for i in ranked_indices
        ]


def answer(question, retriever):
    results = retriever.retrieve(question, k=3)

    context = "\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in results
    )

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    response = mock_llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return {
        "answer": response.choices[0].message.content,
        "sources": [item["source"] for item in results],
        "retrieval": results,
    }


def evaluate(retriever, label):
    evaluation = pd.read_csv(EVAL_PATH, keep_default_na=False)

    required = {"question", "expected_doc", "expected_fact"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError(
            f"Unexpected policy_eval.csv columns: {evaluation.columns.tolist()}. "
            f"Missing expected columns: {sorted(missing)}"
        )

    rows = []

    for _, item in evaluation.iterrows():
        question = item["question"]
        expected_doc = item["expected_doc"].strip()
        expected_fact = item["expected_fact"]
        result = answer(question, retriever)
        answer_text = result["answer"]

        # The two questions without expected documents are unanswerable.
        is_answerable = bool(expected_doc and expected_fact)

        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answerable": is_answerable,
            "retrieved_sources": ";".join(result["sources"]),
            "answer": answer_text,
            "expected_doc_retrieved": (
                expected_doc in result["sources"] if is_answerable else ""
            ),
            "expected_fact_found": (
                expected_fact in answer_text if is_answerable else ""
            ),
            "fallback_correct": (
                answer_text.startswith("I could not find")
                if not is_answerable else ""
            ),
        })

    details = pd.DataFrame(rows)

    answerable = details[details["answerable"]]
    unanswerable = details[~details["answerable"]]

    summary = {
        "run": label,
        "chunks": len(retriever.chunks),
        "answerable_questions": len(answerable),
        "expected_doc_in_top_3": int(
            answerable["expected_doc_retrieved"].sum()
        ),
        "expected_fact_found": int(
            answerable["expected_fact_found"].sum()
        ),
        "unanswerable_questions": len(unanswerable),
        "correct_fallbacks": int(
            unanswerable["fallback_correct"].sum()
        ),
    }

    return summary, details


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retriever = PolicyRetriever(overlap=10)

    print("CHUNK COUNTS")
    counts = {}
    for chunk in retriever.chunks:
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")
    print(f"TOTAL: {len(retriever.chunks)}")

    assert len(retriever.chunks) == 15, (
        f"Expected 15 chunks, got {len(retriever.chunks)}"
    )

    print("\nSAMPLE RETRIEVAL")
    for item in retriever.retrieve("refund policy late delivery", k=3):
        print(
            f"{item['source']} | score={item['score']:.4f} | "
            f"{item['text'][:100]}..."
        )

    summary_overlap, details_overlap = evaluate(
        retriever, "overlap_10"
    )
    details_overlap.to_csv(
        OUTPUT_DIR / "policy_eval_results.csv", index=False
    )

    # Repeat the evaluation with 40-word chunks and zero overlap.
    retriever_zero = PolicyRetriever(overlap=0)
    summary_zero, details_zero = evaluate(
        retriever_zero, "overlap_0"
    )
    details_zero.to_csv(
        OUTPUT_DIR / "policy_eval_zero_overlap.csv", index=False
    )

    summary_df = pd.DataFrame([summary_overlap, summary_zero])
    summary_df.to_csv(
        OUTPUT_DIR / "policy_eval_summary.csv", index=False
    )

    print("\nEVALUATION SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nZERO-OVERLAP ANSWERABLE FAILURES")
    failures = details_zero[
        (details_zero["answerable"])
        & (
            (details_zero["expected_doc_retrieved"] == False)
            | (details_zero["expected_fact_found"] == False)
        )
    ]

    if failures.empty:
        print("No failures found in this run.")
    else:
        for _, row in failures.iterrows():
            print(f"Question: {row['question']}")
            print(f"Expected document: {row['expected_doc']}")
            print(f"Retrieved: {row['retrieved_sources']}")
            print(f"Expected fact: {row['expected_fact']}")
            print(f"Answer: {row['answer']}")
            print()

    print("\nSaved evaluation CSV files in outputs/.")


if __name__ == "__main__":
    main()
