
import json
from pathlib import Path
from collections import Counter


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ramdocs_test.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ramdocs_test_processed.jsonl"


# Original dataset labels
DOCUMENT_LABELS = {
    "correct": "supports",
    "misinfo": "unverified_misinformation",
    "noise": "irrelevant"
}


def load_raw_data():

    records = []

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} raw records.")

    return records


def normalize_record(record, index):

    passages = []

    for passage_index, document in enumerate(
        record["documents"], start=1
    ):

        original_type = document["type"]

        if original_type not in DOCUMENT_LABELS:
            raise ValueError(
                f"Unknown document type: {original_type}"
            )

        passages.append({
            "passage_id": f"P{passage_index}",
            "text": document["text"],
            "original_type": original_type,
            "evidence_label": DOCUMENT_LABELS[original_type],
            "answer": document.get("answer", "unknown")
        })

    normalized = {
        "sample_id": f"ramdocs_{index:04d}",
        "question": record["question"],
        "passages": passages,
        "gold_answers": record["gold_answers"],
        "wrong_answers": record["wrong_answers"],
        "disambig_entity": record.get("disambig_entity"),
        "dataset": "RAMDocs",
        "split": "test",
        "annotation_status": "partially_normalized",
        "evidence_state": None,
        "target_action": None
    }

    return normalized


def validate_record(record):

    required_fields = [
        "sample_id",
        "question",
        "passages",
        "gold_answers",
        "wrong_answers"
    ]

    for field in required_fields:

        if field not in record:
            raise ValueError(f"Missing field: {field}")

    for passage in record["passages"]:

        if passage["evidence_label"] not in [
            "supports",
            "unverified_misinformation",
            "irrelevant"
        ]:
            raise ValueError("Invalid passage label")

    return True


def save_processed_data(records):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            validate_record(record)

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"\nProcessed data saved: {OUTPUT_FILE}")


def main():

    print("=" * 60)
    print("VERDICT - RAMDocs Processing")
    print("=" * 60)

    raw_records = load_raw_data()

    processed_records = []

    label_counts = Counter()

    for index, record in enumerate(raw_records, start=1):

        normalized = normalize_record(record, index)

        validate_record(normalized)

        processed_records.append(normalized)

        for passage in normalized["passages"]:
            label_counts[passage["evidence_label"]] += 1

    print("\nEVIDENCE ANNOTATION DISTRIBUTION")

    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\nTOTAL PROCESSED RECORDS:", len(processed_records))

    save_processed_data(processed_records)

    print("\nRAMDocs processing completed successfully!")


if __name__ == "__main__":
    main()
