
import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path(
    "data/processed/ramdocs_abstention_decisions.jsonl"
)

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))


print("=" * 75)
print("VERDICT - Abstention Decision Inspection")
print("=" * 75)

decision_counts = Counter(
    r["abstention"]["decision"] for r in records
)

state_counts = Counter(
    (
        r.get("aggregation", {}).get("evidence_state"),
        r["abstention"]["decision"]
    )
    for r in records
)

reason_counts = Counter(
    r["abstention"]["abstention_reason"]
    for r in records
    if r["abstention"]["abstention_reason"]
)

print("\nDECISION DISTRIBUTION")

for decision, count in decision_counts.items():
    print(f"{decision}: {count}")


print("\nEVIDENCE STATE AND DECISION")

for (state, decision), count in sorted(state_counts.items()):
    print(f"{state:15} -> {decision:10}: {count}")


print("\nABSTENTION REASONS")

for reason, count in reason_counts.items():
    print(f"{reason}: {count}")


print("\n" + "=" * 75)
print("ANSWERED EXAMPLES")
print("=" * 75)

answered = [
    r for r in records
    if r["abstention"]["decision"] == "answer"
]

for record in answered[:5]:

    print("\nQuestion:", record["question"])
    print("Candidate:", record.get("candidate_answer"))
    print("Final answer:", record["abstention"]["final_answer"])
    print("Evidence state:",
          record["aggregation"]["evidence_state"])
    print("Abstention score:",
          record["abstention"]["abstention_score"])


print("\n" + "=" * 75)
print("ABSTAINED EXAMPLES")
print("=" * 75)

abstained = [
    r for r in records
    if r["abstention"]["decision"] == "abstain"
]

for record in abstained[:5]:

    print("\nQuestion:", record["question"])
    print("Candidate:", record.get("candidate_answer"))
    print("Evidence state:",
          record["aggregation"]["evidence_state"])
    print("Reason:",
          record["abstention"]["abstention_reason"])
