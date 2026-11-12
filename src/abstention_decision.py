
import json
from pathlib import Path
from tqdm import tqdm

INPUT_FILE = Path(
    "data/processed/ramdocs_aggregated_evidence.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/ramdocs_abstention_decisions.jsonl"
)

ABSTENTION_THRESHOLD = 0.50

# Initial heuristic scores.
ABSTENTION_SCORES = {
    "consistent_support": 0.10,
    "consistent_refutation": 0.95,
    "contradictory": 0.95,
    "ambiguous": 0.65,
    "insufficient": 0.99
}


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
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def make_decision(record):

    aggregation = record.get("aggregation", {})

    evidence_state = aggregation.get(
        "evidence_state", "insufficient"
    )

    dominant_relation = aggregation.get(
        "dominant_relation"
    )

    candidate_answer = record.get("candidate_answer")

    # Determine abstention score and explanation.
    if (
        evidence_state == "consistent"
        and dominant_relation == "supports"
        and candidate_answer
    ):

        score = ABSTENTION_SCORES["consistent_support"]

        reason = "Strong and consistent supporting evidence."

    elif (
        evidence_state == "consistent"
        and dominant_relation == "refutes"
    ):

        score = ABSTENTION_SCORES["consistent_refutation"]

        reason = "Available evidence consistently refutes the candidate answer."

    elif evidence_state == "contradictory":

        score = ABSTENTION_SCORES["contradictory"]

        reason = "Supporting and refuting evidence conflict."

    elif evidence_state == "ambiguous":

        score = ABSTENTION_SCORES["ambiguous"]

        reason = "Evidence is uncertain or insufficiently decisive."

    else:

        score = ABSTENTION_SCORES["insufficient"]

        reason = "Insufficient usable evidence or missing candidate answer."

    # Apply abstention threshold.
    if score < ABSTENTION_THRESHOLD:

        action = "answer"
        final_answer = candidate_answer

    else:

        action = "abstain"
        final_answer = None

    return {
        "abstention_score": score,
        "abstention_threshold": ABSTENTION_THRESHOLD,
        "decision": action,
        "final_answer": final_answer,
        "abstention_reason": (
            None if action == "answer" else reason
        ),
        "decision_method": "rule_based_baseline"
    }


def main():

    print("=" * 70)
    print("VERDICT - Evidence-Aware Abstention")
    print("=" * 70)

    records = load_jsonl(INPUT_FILE)

    print(f"\nQuestions loaded: {len(records)}")

    output_records = []

    for record in tqdm(records, desc="Making decisions"):

        decision = make_decision(record)

        updated_record = {
            **record,
            "abstention": decision
        }

        output_records.append(updated_record)

    save_jsonl(output_records, OUTPUT_FILE)

    answers = sum(
        r["abstention"]["decision"] == "answer"
        for r in output_records
    )

    abstentions = len(output_records) - answers

    total = len(output_records)

    coverage = answers / total * 100 if total else 0
    abstention_rate = abstentions / total * 100 if total else 0

    print("\n" + "=" * 70)
    print("ABSTENTION RESULTS")
    print("=" * 70)

    print(f"Total questions: {total}")
    print(f"Answers generated: {answers}")
    print(f"Abstentions: {abstentions}")

    print(f"Coverage: {coverage:.2f}%")
    print(f"Abstention rate: {abstention_rate:.2f}%")

    print("\nOutput file:")
    print(OUTPUT_FILE.resolve())

    print("\nAbstention decision completed!")


if __name__ == "__main__":
    main()
