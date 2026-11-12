
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ramdocs_retrieved.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ramdocs_screened.jsonl"
)

# Frozen using the development-set calibration experiment.
SCREENING_THRESHOLD = 0.325


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_records(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def screen_passages(records):
    screened_records = []

    total_passages = 0
    retained_passages = 0
    removed_passages = 0

    for record in records:
        retained = []
        removed = []

        for passage in record["retrieved_passages"]:
            total_passages += 1

            score = float(passage["similarity_score"])

            # Screening decision uses similarity only.
            if score >= SCREENING_THRESHOLD:
                retained.append(passage)
                retained_passages += 1
            else:
                removed.append(passage)
                removed_passages += 1

        screened_record = dict(record)

        screened_record["retained_passages"] = retained
        screened_record["screened_out_passages"] = removed

        screened_record["screening_metadata"] = {
            "method": "cosine_similarity_threshold",
            "threshold": SCREENING_THRESHOLD,
            "total_passages": len(record["retrieved_passages"]),
            "retained_count": len(retained),
            "removed_count": len(removed)
        }

        screened_records.append(screened_record)

    return (
        screened_records,
        total_passages,
        retained_passages,
        removed_passages
    )


def main():
    print("=" * 65)
    print("VERDICT - Passage Screening")
    print("=" * 65)

    records = load_records()

    (
        screened_records,
        total,
        retained,
        removed
    ) = screen_passages(records)

    save_records(screened_records)

    print("\nSCREENING CONFIGURATION")
    print("-" * 65)
    print("Method:", "Cosine similarity threshold")
    print("Threshold:", SCREENING_THRESHOLD)

    print("\nSCREENING RESULTS")
    print("-" * 65)
    print("Questions processed:", len(records))
    print("Total retrieved passages:", total)
    print("Retained passages:", retained)
    print("Removed passages:", removed)

    if total:
        print(
            "Retention rate:",
            f"{retained / total * 100:.2f}%"
        )
        print(
            "Removal rate:",
            f"{removed / total * 100:.2f}%"
        )

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nPassage screening completed successfully!")


if __name__ == "__main__":
    main()
