
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ramdocs_development.jsonl"
)

MIN_RELEVANT_RECALL = 0.95


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    print("=" * 70)
    print("VERDICT - Screening Threshold Calibration")
    print("=" * 70)

    records = load_records()

    passages = []

    for record in records:
        for passage in record["retrieved_passages"]:
            passages.append({
                "score": float(passage["similarity_score"]),
                "label": passage["original_type"]
            })

    relevant_labels = {"correct", "misinfo"}

    total_relevant = sum(
        p["label"] in relevant_labels for p in passages
    )

    total_noise = sum(
        p["label"] == "noise" for p in passages
    )

    print("\nDevelopment questions:", len(records))
    print("Total retrieved passages:", len(passages))
    print("Relevant passages:", total_relevant)
    print("Noise passages:", total_noise)

    results = []

    print("\nTHRESHOLD ANALYSIS")
    print("-" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Relevant Recall':<20}"
        f"{'Noise Removal':<18}"
        f"{'Keep Precision':<18}"
    )

    threshold = 0.05

    while threshold <= 0.750001:
        kept = [
            p for p in passages
            if p["score"] >= threshold
        ]

        kept_relevant = sum(
            p["label"] in relevant_labels for p in kept
        )

        kept_noise = sum(
            p["label"] == "noise" for p in kept
        )

        relevant_recall = (
            kept_relevant / total_relevant
            if total_relevant else 0
        )

        noise_removal = (
            (total_noise - kept_noise) / total_noise
            if total_noise else 0
        )

        keep_precision = (
            kept_relevant / len(kept)
            if kept else 0
        )

        results.append({
            "threshold": round(threshold, 3),
            "relevant_recall": relevant_recall,
            "noise_removal": noise_removal,
            "keep_precision": keep_precision,
            "kept_passages": len(kept)
        })

        print(
            f"{threshold:<12.2f}"
            f"{relevant_recall:<20.4f}"
            f"{noise_removal:<18.4f}"
            f"{keep_precision:<18.4f}"
        )

        threshold += 0.025

    eligible = [
        result for result in results
        if result["relevant_recall"] >= MIN_RELEVANT_RECALL
    ]

    print("\n" + "=" * 70)

    if eligible:
        # Highest threshold that preserves at least 95% of relevant evidence.
        best = max(eligible, key=lambda x: x["threshold"])

        print("RECOMMENDED DEVELOPMENT THRESHOLD")
        print("-" * 70)
        print("Threshold:", best["threshold"])
        print("Relevant recall:", round(best["relevant_recall"], 4))
        print("Noise removal:", round(best["noise_removal"], 4))
        print("Keep precision:", round(best["keep_precision"], 4))
        print("Passages retained:", best["kept_passages"])
        print(
            "Relevant recall requirement:",
            MIN_RELEVANT_RECALL
        )
    else:
        print("No threshold satisfies the relevant recall requirement.")
        print("Use a lower threshold or reconsider the screening approach.")

    print("\nCalibration completed successfully!")


if __name__ == "__main__":
    main()
