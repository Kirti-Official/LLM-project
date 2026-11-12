
import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data/raw/ramdocs_test.jsonl"

RETRIEVED_FILE = (
    PROJECT_ROOT / "data/processed/ramdocs_retrieved.jsonl"
)


def load_jsonl(file_path):

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def evaluate_retrieval():

    print("=" * 60)
    print("VERDICT - Retrieval Evaluation")
    print("=" * 60)

    raw_records = load_jsonl(RAW_FILE)
    retrieved_records = load_jsonl(RETRIEVED_FILE)

    if len(raw_records) != len(retrieved_records):
        raise ValueError("Raw and retrieved record counts differ.")

    total_retrieved = 0
    correct_retrieved = 0
    misinfo_retrieved = 0
    noise_retrieved = 0
    questions_with_correct = 0

    category_counts = Counter()

    for raw, retrieved in zip(
        raw_records, retrieved_records
    ):

        passages = retrieved["retrieved_passages"]

        total_retrieved += len(passages)

        has_correct = False

        for passage in passages:

            label = passage["original_type"]

            category_counts[label] += 1

            if label == "correct":
                correct_retrieved += 1
                has_correct = True

            elif label == "misinfo":
                misinfo_retrieved += 1

            elif label == "noise":
                noise_retrieved += 1

        if has_correct:
            questions_with_correct += 1

    total_questions = len(raw_records)

    precision_at_k = (
        correct_retrieved / total_retrieved
        if total_retrieved else 0
    )

    hit_at_k = (
        questions_with_correct / total_questions
        if total_questions else 0
    )

    print("\n1. RETRIEVAL COUNTS")
    print("Questions:", total_questions)
    print("Retrieved passages:", total_retrieved)

    print("\n2. PASSAGE DISTRIBUTION")

    for category in ["correct", "misinfo", "noise"]:

        count = category_counts[category]

        percentage = (
            count / total_retrieved * 100
            if total_retrieved else 0
        )

        print(
            f"{category}: {count} ({percentage:.2f}%)"
        )

    print("\n3. RETRIEVAL METRICS")

    print(
        "Precision@3:",
        f"{precision_at_k:.4f}"
    )

    print(
        "Hit@3:",
        f"{hit_at_k:.4f}"
    )

    print(
        "Questions with correct evidence:",
        questions_with_correct
    )

    print(
        "Misinformation exposure:",
        f"{misinfo_retrieved / total_retrieved * 100:.2f}%"
    )

    print(
        "Noise exposure:",
        f"{noise_retrieved / total_retrieved * 100:.2f}%"
    )

    print("\nRetrieval evaluation completed!")


if __name__ == "__main__":
    evaluate_retrieval()
