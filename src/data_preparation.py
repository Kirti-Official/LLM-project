
import json
from pathlib import Path

# Allowed VERDICT labels
EVIDENCE_LABELS = {
    "supports",
    "refutes",
    "irrelevant"
}

EVIDENCE_STATES = {
    "consistent",
    "ambiguous",
    "contradictory",
    "insufficient"
}

TARGET_ACTIONS = {
    "answer",
    "alternatives",
    "abstain"
}


def validate_record(record):
    """Validate one normalized VERDICT example."""

    required_fields = [
        "sample_id",
        "question",
        "passages",
        "gold_answers",
        "evidence_state",
        "target_action"
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing field: {field}")

    if not isinstance(record["passages"], list):
        raise ValueError("Passages must be a list")

    if not isinstance(record["gold_answers"], list):
        raise ValueError("Gold answers must be a list")

    for passage in record["passages"]:
        if not passage.get("passage_id"):
            raise ValueError("Missing passage ID")

        if not passage.get("text"):
            raise ValueError("Missing passage text")

        label = passage.get("evidence_label")

        if label not in EVIDENCE_LABELS:
            raise ValueError(f"Invalid evidence label: {label}")

    if record["evidence_state"] not in EVIDENCE_STATES:
        raise ValueError("Invalid evidence state")

    if record["target_action"] not in TARGET_ACTIONS:
        raise ValueError("Invalid target action")

    return True


def load_jsonl(file_path):
    """Load normalized examples from a JSONL file."""

    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            try:
                validate_record(record)
            except ValueError as error:
                raise ValueError(
                    f"Error on line {line_number}: {error}"
                ) from error

            records.append(record)

    return records


def save_jsonl(records, file_path):
    """Save normalized examples to JSONL."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    for record in records:
        validate_record(record)

    with open(file_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {file_path}")


if __name__ == "__main__":

    print("VERDICT Data Preparation Module")
    print("Evidence labels:", EVIDENCE_LABELS)
    print("Evidence states:", EVIDENCE_STATES)
    print("Target actions:", TARGET_ACTIONS)
    print("Module initialized successfully!")
