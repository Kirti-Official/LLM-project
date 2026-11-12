
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOLDOUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_exploratory_holdout.jsonl"
)

SCREENED_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_screened.jsonl"
)

RELEVANT_LABELS = {"correct", "misinfo"}


def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_record_key(record):
    return record.get("sample_id", record["question"])


def main():
    print("=" * 65)
    print("VERDICT - Exploratory Holdout Screening Evaluation")
    print("=" * 65)

    holdout_records = load_jsonl(HOLDOUT_FILE)
    screened_records = load_jsonl(SCREENED_FILE)

    screened_map = {
        get_record_key(record): record
        for record in screened_records
    }

    total = 0
    retained = 0
    removed = 0

    relevant_total = 0
    relevant_retained = 0

    noise_total = 0
    noise_retained = 0

    for original_record in holdout_records:
        key = get_record_key(original_record)

        if key not in screened_map:
            raise ValueError(f"Missing screened record: {key}")

        screened_record = screened_map[key]

        original_passages = original_record["retrieved_passages"]
        retained_passages = screened_record["retained_passages"]

        retained_ids = {
            passage["passage_id"]
            for passage in retained_passages
        }

        for passage in original_passages:
            total += 1

            label = passage["original_type"]
            is_retained = passage["passage_id"] in retained_ids

            if is_retained:
                retained += 1
            else:
                removed += 1

            if label in RELEVANT_LABELS:
                relevant_total += 1

                if is_retained:
                    relevant_retained += 1

            elif label == "noise":
                noise_total += 1

                if is_retained:
                    noise_retained += 1

    relevant_recall = (
        relevant_retained / relevant_total
        if relevant_total else 0
    )

    noise_removal = (
        (noise_total - noise_retained) / noise_total
        if noise_total else 0
    )

    keep_precision = (
        relevant_retained / retained
        if retained else 0
    )

    print("\nDATASET")
    print("-" * 65)
    print("Holdout questions:", len(holdout_records))
    print("Total passages:", total)

    print("\nSCREENING RESULTS")
    print("-" * 65)
    print("Retained:", retained)
    print("Removed:", removed)

    print("\nANNOTATION-BASED EVALUATION")
    print("-" * 65)
    print("Relevant passages:", relevant_total)
    print("Relevant passages retained:", relevant_retained)
    print("Noise passages:", noise_total)
    print("Noise passages retained:", noise_retained)

    print("\nMETRICS")
    print("-" * 65)
    print(f"Relevant recall: {relevant_recall:.4f}")
    print(f"Noise removal: {noise_removal:.4f}")
    print(f"Keep precision: {keep_precision:.4f}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
