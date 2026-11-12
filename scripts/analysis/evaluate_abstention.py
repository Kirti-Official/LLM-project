
import json
import re
import string
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/ramdocs_abstention_decisions.jsonl"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records


def normalize(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Normalize numeric formatting.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # Remove punctuation.
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Normalize whitespace.
    text = " ".join(text.split())

    return text


def to_list(value):

    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value if x is not None]

    return [str(value)]


def matches_gold(candidate, gold_answers):

    candidate_norm = normalize(candidate)

    if not candidate_norm:
        return False

    return any(
        candidate_norm == normalize(answer)
        for answer in gold_answers
    )


def matches_wrong(candidate, wrong_answers):

    candidate_norm = normalize(candidate)

    if not candidate_norm:
        return False

    return any(
        candidate_norm == normalize(answer)
        for answer in wrong_answers
    )


def main():

    records = load_jsonl(INPUT_FILE)

    total = len(records)

    answered_records = [
        r for r in records
        if r["abstention"]["decision"] == "answer"
    ]

    abstained_records = [
        r for r in records
        if r["abstention"]["decision"] == "abstain"
    ]

    correct_answers = []
    wrong_answers = []
    unmatched_answers = []

    false_abstentions = []
    missing_candidates = []

    # Evaluate accepted answers.
    for record in answered_records:

        candidate = record.get("candidate_answer")

        gold = to_list(record.get("gold_answers"))
        wrong = to_list(record.get("wrong_answers"))

        if not candidate:
            unmatched_answers.append(record)
            continue

        if matches_gold(candidate, gold):
            correct_answers.append(record)

        elif matches_wrong(candidate, wrong):
            wrong_answers.append(record)

        else:
            unmatched_answers.append(record)

    # Check whether abstained candidates were actually correct.
    for record in abstained_records:

        candidate = record.get("candidate_answer")

        gold = to_list(record.get("gold_answers"))

        if not candidate:
            missing_candidates.append(record)

        elif matches_gold(candidate, gold):
            false_abstentions.append(record)

    answered = len(answered_records)
    abstained = len(abstained_records)

    correct = len(correct_answers)
    wrong = len(wrong_answers)
    unmatched = len(unmatched_answers)

    coverage = answered / total if total else 0

    selective_accuracy = correct / answered if answered else 0

    selective_risk = 1 - selective_accuracy if answered else 0

    false_abstention_rate = (
        len(false_abstentions) / abstained
        if abstained else 0
    )

    wrong_answer_acceptance = (
        wrong / answered if answered else 0
    )

    print("=" * 75)
    print("VERDICT - Abstention Evaluation")
    print("=" * 75)

    print(f"\nTotal questions: {total}")
    print(f"Answered: {answered}")
    print(f"Abstained: {abstained}")

    print("\nANSWER QUALITY")
    print("-" * 75)

    print(f"Exact-match correct: {correct}")
    print(f"Matched wrong answers: {wrong}")
    print(f"Unmatched answers: {unmatched}")

    print(f"\nCoverage: {coverage * 100:.2f}%")
    print(
        f"Selective accuracy: "
        f"{selective_accuracy * 100:.2f}%"
    )
    print(
        f"Selective risk: "
        f"{selective_risk * 100:.2f}%"
    )
    print(
        f"Wrong-answer acceptance rate: "
        f"{wrong_answer_acceptance * 100:.2f}%"
    )

    print("\nABSTENTION QUALITY")
    print("-" * 75)

    print(
        f"Correct candidates unnecessarily rejected: "
        f"{len(false_abstentions)}"
    )

    print(
        f"False abstention rate among abstentions: "
        f"{false_abstention_rate * 100:.2f}%"
    )

    print(
        f"Abstentions with missing candidates: "
        f"{len(missing_candidates)}"
    )

    print("\n" + "=" * 75)
    print("ACCEPTED ANSWER DETAILS")
    print("=" * 75)

    for record in answered_records:

        print("\nQuestion:", record["question"])
        print("Candidate:", record.get("candidate_answer"))
        print("Gold answers:", record.get("gold_answers"))
        print("Wrong answers:", record.get("wrong_answers"))

        if record in correct_answers:
            print("Evaluation: CORRECT")
        elif record in wrong_answers:
            print("Evaluation: MATCHED WRONG ANSWER")
        else:
            print("Evaluation: UNMATCHED")

    print("\n" + "=" * 75)
    print("FALSE ABSTENTION EXAMPLES")
    print("=" * 75)

    for record in false_abstentions[:10]:

        print("\nQuestion:", record["question"])
        print("Candidate:", record.get("candidate_answer"))
        print("Gold answers:", record.get("gold_answers"))
        print("Evidence state:",
              record["aggregation"]["evidence_state"])

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
