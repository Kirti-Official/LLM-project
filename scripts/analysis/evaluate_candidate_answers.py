
import json
import re
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_with_candidates.jsonl"
)


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction, references):
    prediction = normalize(prediction)

    return any(
        prediction == normalize(reference)
        for reference in references
    )


def token_f1(prediction, reference):
    prediction_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction, references):
    if not references:
        return 0.0

    return max(
        token_f1(prediction, reference)
        for reference in references
    )


def main():
    print("=" * 70)
    print("VERDICT - Candidate Answer Evaluation")
    print("=" * 70)

    records = load_records()

    total = len(records)
    generated = 0
    exact_correct = 0
    wrong_matches = 0
    f1_scores = []

    for record in records:

        prediction = record.get("candidate_answer")

        gold_answers = record.get("gold_answers") or []
        wrong_answers = record.get("wrong_answers") or []

        if not prediction:
            continue

        generated += 1

        if exact_match(prediction, gold_answers):
            exact_correct += 1

        if exact_match(prediction, wrong_answers):
            wrong_matches += 1

        if gold_answers:
            f1_scores.append(
                best_token_f1(prediction, gold_answers)
            )

    coverage = generated / total if total else 0

    exact_match_rate = (
        exact_correct / generated if generated else 0
    )

    wrong_match_rate = (
        wrong_matches / generated if generated else 0
    )

    average_f1 = (
        sum(f1_scores) / len(f1_scores)
        if f1_scores else 0
    )

    print("\nDATASET")
    print("-" * 70)
    print("Total questions:", total)
    print("Generated answers:", generated)
    print("Missing answers:", total - generated)

    print("\nEVALUATION METRICS")
    print("-" * 70)
    print(f"Generation coverage: {coverage:.4f}")
    print(f"Exact Match rate: {exact_match_rate:.4f}")
    print(f"Average Token F1: {average_f1:.4f}")
    print(f"Wrong Answer Match rate: {wrong_match_rate:.4f}")

    print("\nCOUNTS")
    print("-" * 70)
    print("Exact gold matches:", exact_correct)
    print("Exact wrong-answer matches:", wrong_matches)
    print("Answers evaluated with gold references:", len(f1_scores))

    print("\nFIRST TEN EXAMPLES")
    print("-" * 70)

    for record in records[:10]:
        print("\nQuestion:", record["question"])
        print("Candidate:", record.get("candidate_answer"))
        print("Gold:", record.get("gold_answers"))
        print("Wrong:", record.get("wrong_answers"))

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
