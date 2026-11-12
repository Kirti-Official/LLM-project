
import json
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/ramdocs_evidence_predictions.jsonl"
)

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))

print("=" * 75)
print("VERDICT - Evidence Prediction Inspection")
print("=" * 75)

# Find examples from each predicted class
target_labels = ["supports", "refutes", "irrelevant"]

for target in target_labels:

    print(f"\n{'=' * 75}")
    print(f"EXAMPLES: {target.upper()}")
    print("=" * 75)

    shown = 0

    for record in records:

        candidate = record.get("candidate_answer")

        if not candidate:
            continue

        predictions = record.get("evidence_predictions", [])

        for prediction in predictions:

            if prediction["predicted_label"] != target:
                continue

            print(f"\nQuestion: {record['question']}")
            print(f"Candidate answer: {candidate}")
            print(f"Passage ID: {prediction['passage_id']}")
            print(f"Predicted label: {prediction['predicted_label']}")

            print("Probabilities:")
            for label, score in prediction["probabilities"].items():
                print(f"  {label}: {score:.4f}")

            # Find the matching retrieved passage
            passages = record.get("retained_passages", [])

            for passage in passages:
                if passage.get("passage_id") == prediction["passage_id"]:
                    print("Passage:")
                    print(passage["text"][:700])
                    break

            shown += 1

            if shown == 3:
                break

        if shown == 3:
            break
