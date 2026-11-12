
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ramdocs_retrieved.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42
DEVELOPMENT_RATIO = 0.80


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_records(records, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("VERDICT - Development Dataset Split")
    print("=" * 60)

    records = load_records()

    # Shuffle complete questions, not individual passages.
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(records)

    split_index = int(len(records) * DEVELOPMENT_RATIO)

    development_records = records[:split_index]
    exploratory_holdout_records = records[split_index:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    development_file = OUTPUT_DIR / "ramdocs_development.jsonl"
    holdout_file = OUTPUT_DIR / "ramdocs_exploratory_holdout.jsonl"

    save_records(development_records, development_file)
    save_records(exploratory_holdout_records, holdout_file)

    print("\nDataset split completed!")
    print("Total questions:", len(records))
    print("Development questions:", len(development_records))
    print("Exploratory holdout questions:", len(exploratory_holdout_records))

    print("\nSaved files:")
    print(development_file)
    print(holdout_file)

    print("\nRandom seed:", RANDOM_SEED)
    print("Development ratio:", DEVELOPMENT_RATIO)

    print("\nSplit completed successfully!")


if __name__ == "__main__":
    main()
