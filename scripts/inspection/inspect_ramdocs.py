
from datasets import load_dataset
from collections import Counter
from pathlib import Path
import json


def inspect_ramdocs():

    print("=" * 60)
    print("VERDICT - RAMDocs Dataset Inspection")
    print("=" * 60)

    # 1. Load official RAMDocs test split
    print("\nLoading RAMDocs dataset...")

    dataset = load_dataset(
        "HanNight/RAMDocs",
        split="test"
    )

    print("\n1. DATASET INFORMATION")
    print("Total records:", len(dataset))
    print("Columns:", dataset.column_names)

    # 2. Inspect document categories
    document_types = Counter()
    total_passages = 0

    for record in dataset:
        for document in record["documents"]:
            document_types[document["type"]] += 1
            total_passages += 1

    print("\n2. DOCUMENT TYPE DISTRIBUTION")

    for doc_type, count in document_types.items():
        percentage = (count / total_passages) * 100

        print(
            f"{doc_type}: {count} "
            f"({percentage:.2f}%)"
        )

    print("Total passages:", total_passages)

    # 3. Inspect first record
    print("\n3. FIRST RECORD")

    first_record = dataset[0]

    print("Question:", first_record["question"])

    print("\nDocuments:")

    for index, document in enumerate(
        first_record["documents"], start=1
    ):

        print(f"\nPassage {index}")
        print("Type:", document["type"])
        print("Answer:", document["answer"])
        print("Text:", document["text"][:300])

    print("\nGold answers:", first_record["gold_answers"])
    print("Wrong answers:", first_record["wrong_answers"])

    # 4. Create output directory automatically
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 5. Save original dataset without changing annotations
    output_path = raw_dir / "ramdocs_test.jsonl"

    print("\nSaving raw dataset...")

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            for record in dataset:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

        print("\n4. RAW DATA SAVED")
        print("Location:", output_path)
        print("Records saved:", len(dataset))
        print(
            "File size:",
            f"{output_path.stat().st_size / 1024:.2f} KB"
        )

    except OSError as error:

        print("\nERROR WHILE SAVING DATASET")
        print(error)
        return

    print("\nDataset inspection completed successfully!")


if __name__ == "__main__":
    inspect_ramdocs()
