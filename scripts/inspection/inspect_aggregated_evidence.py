
import json
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/ramdocs_aggregated_evidence.jsonl"
)

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))


def display_record(record):

    aggregation = record["aggregation"]

    print("\n" + "=" * 75)
    print("Question:", record["question"])
    print("Candidate answer:", record.get("candidate_answer"))

    print("\nEvidence state:", aggregation["evidence_state"])
    print("Dominant relation:", aggregation["dominant_relation"])

    print("\nPassage counts:")
    print("Supports:", aggregation["support_count"])
    print("Refutes:", aggregation["refute_count"])
    print("Irrelevant:", aggregation["irrelevant_count"])

    print("\nStrong evidence:")
    print("Strong supports:", aggregation["strong_support_count"])
    print("Strong refutes:", aggregation["strong_refute_count"])

    print("\nMaximum probabilities:")
    print("Support:", aggregation.get("max_support_probability"))
    print("Refute:", aggregation.get("max_refute_probability"))

    print("\nSupporting passage IDs:",
          aggregation["support_passage_ids"])

    print("Refuting passage IDs:",
          aggregation["refute_passage_ids"])


# 1. Inspect one example from every evidence state.

print("=" * 75)
print("EXAMPLES BY EVIDENCE STATE")
print("=" * 75)

states = [
    "consistent",
    "contradictory",
    "ambiguous",
    "insufficient"
]

for state in states:

    matching = [
        record for record in records
        if record["aggregation"]["evidence_state"] == state
    ]

    print(f"\n\nSTATE: {state.upper()}")
    print("Available examples:", len(matching))

    for record in matching[:2]:
        display_record(record)


# 2. Inspect important diagnostic questions.

target_questions = [
    "What is the population of Broken Bow?",
    "What sport is Ryan Davis associated with?",
    'Who are the directors of the film "Lahu Ke Do Rang"?',
    "What sport is Bobby Carpenter associated with?"
]

print("\n\n" + "=" * 75)
print("TARGETED VERDICT VALIDATION")
print("=" * 75)

for question in target_questions:

    matching = [
        record for record in records
        if record["question"] == question
    ]

    if matching:
        display_record(matching[0])
    else:
        print("\nQuestion not found:", question)
