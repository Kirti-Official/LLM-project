
import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = Path(
    "data/processed/ramdocs_evidence_predictions.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/ramdocs_aggregated_evidence.jsonl"
)

# Provisional experimental threshold.
STRONG_THRESHOLD = 0.80
WEAK_THRESHOLD = 0.50


def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def save_jsonl(records, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def aggregate_evidence(record):

    predictions = record.get("evidence_predictions", [])

    if not record.get("candidate_answer") or not predictions:

        return {
            "evidence_state": "insufficient",
            "dominant_relation": None,
            "support_count": 0,
            "refute_count": 0,
            "irrelevant_count": 0,
            "strong_support_count": 0,
            "strong_refute_count": 0,
            "support_passage_ids": [],
            "refute_passage_ids": [],
            "aggregation_threshold": STRONG_THRESHOLD,
            "aggregation_status": "no_usable_evidence"
        }

    support_ids = []
    refute_ids = []
    irrelevant_ids = []

    strong_support_ids = []
    strong_refute_ids = []

    max_support = 0.0
    max_refute = 0.0

    for prediction in predictions:

        probabilities = prediction["probabilities"]

        support_score = probabilities["entailment"]
        refute_score = probabilities["contradiction"]

        passage_id = prediction["passage_id"]

        max_support = max(max_support, support_score)
        max_refute = max(max_refute, refute_score)

        if prediction["predicted_label"] == "supports":
            support_ids.append(passage_id)

        elif prediction["predicted_label"] == "refutes":
            refute_ids.append(passage_id)

        else:
            irrelevant_ids.append(passage_id)

        if support_score >= STRONG_THRESHOLD:
            strong_support_ids.append(passage_id)

        if refute_score >= STRONG_THRESHOLD:
            strong_refute_ids.append(passage_id)

    strong_support = len(strong_support_ids) > 0
    strong_refute = len(strong_refute_ids) > 0

    moderate_support = max_support >= WEAK_THRESHOLD
    moderate_refute = max_refute >= WEAK_THRESHOLD

    # Both directions have strong evidence.
    if strong_support and strong_refute:

        state = "contradictory"
        relation = "conflicting"

    # Strong support without meaningful refutation.
    elif strong_support and not moderate_refute:

        state = "consistent"
        relation = "supports"

    # Strong refutation without meaningful support.
    elif strong_refute and not moderate_support:

        state = "consistent"
        relation = "refutes"

    # Evidence is present but uncertain or conflicting.
    elif moderate_support or moderate_refute:

        state = "ambiguous"
        relation = "uncertain"

    else:

        state = "insufficient"
        relation = None

    return {
        "evidence_state": state,
        "dominant_relation": relation,

        "support_count": len(support_ids),
        "refute_count": len(refute_ids),
        "irrelevant_count": len(irrelevant_ids),

        "strong_support_count": len(strong_support_ids),
        "strong_refute_count": len(strong_refute_ids),

        "support_passage_ids": support_ids,
        "refute_passage_ids": refute_ids,
        "irrelevant_passage_ids": irrelevant_ids,

        "strong_support_passage_ids": strong_support_ids,
        "strong_refute_passage_ids": strong_refute_ids,

        "max_support_probability": round(max_support, 4),
        "max_refute_probability": round(max_refute, 4),

        "aggregation_threshold": STRONG_THRESHOLD,
        "aggregation_status": "aggregated"
    }


def main():

    print("=" * 70)
    print("VERDICT - Evidence Aggregation")
    print("=" * 70)

    records = load_jsonl(INPUT_FILE)

    print(f"\nQuestions loaded: {len(records)}")

    aggregated_records = []

    for record in tqdm(records, desc="Aggregating evidence"):

        aggregation = aggregate_evidence(record)

        updated_record = {
            **record,
            "aggregation": aggregation
        }

        aggregated_records.append(updated_record)

    save_jsonl(aggregated_records, OUTPUT_FILE)

    states = {
        "consistent": 0,
        "contradictory": 0,
        "ambiguous": 0,
        "insufficient": 0
    }

    for record in aggregated_records:

        state = record["aggregation"]["evidence_state"]
        states[state] += 1

    print("\n" + "=" * 70)
    print("AGGREGATION RESULTS")
    print("=" * 70)

    print(f"Total questions: {len(aggregated_records)}")

    for state, count in states.items():

        percentage = (
            count / len(aggregated_records) * 100
            if aggregated_records else 0
        )

        print(
            f"{state.capitalize():15}: "
            f"{count:4} ({percentage:.2f}%)"
        )

    print("\nOutput file:")
    print(OUTPUT_FILE.resolve())

    print("\nEvidence aggregation completed!")


if __name__ == "__main__":
    main()
