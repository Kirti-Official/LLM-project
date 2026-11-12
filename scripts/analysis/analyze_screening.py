
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ramdocs_retrieved.jsonl"
)


def load_records():

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    print("=" * 60)
    print("VERDICT - Passage Screening Analysis")
    print("=" * 60)

    records = load_records()

    scores = defaultdict(list)

    for record in records:

        for passage in record["retrieved_passages"]:

            label = passage["original_type"]

            score = passage["similarity_score"]

            scores[label].append(score)

    print("\nSIMILARITY SCORE DISTRIBUTION")

    for label in ["correct", "misinfo", "noise"]:

        values = scores[label]

        if not values:
            continue

        values = np.array(values)

        print(f"\n{label.upper()}")

        print("Count:", len(values))
        print("Minimum:", round(float(values.min()), 4))
        print("Maximum:", round(float(values.max()), 4))
        print("Mean:", round(float(values.mean()), 4))
        print("Median:", round(float(np.median(values)), 4))
        print(
            "10th percentile:",
            round(float(np.percentile(values, 10)), 4)
        )
        print(
            "90th percentile:",
            round(float(np.percentile(values, 90)), 4)
        )

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
