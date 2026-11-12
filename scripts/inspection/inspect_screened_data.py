
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_screened.jsonl"
)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        first_record = json.loads(next(line for line in file if line.strip()))

    print("=" * 65)
    print("VERDICT - Screened Data Inspection")
    print("=" * 65)

    print("\nRECORD FIELDS")
    for key in first_record.keys():
        print("-", key)

    print("\nQUESTION")
    print(first_record.get("question"))

    print("\nGOLD ANSWERS")
    print(first_record.get("gold_answers"))

    print("\nRETAINED PASSAGES")

    for index, passage in enumerate(
        first_record.get("retained_passages", [])[:3],
        start=1
    ):
        print(f"\nPassage {index}")
        print("Passage ID:", passage.get("passage_id"))
        print("Similarity:", passage.get("similarity_score"))
        print("Text:", str(passage.get("text", ""))[:600])
        print("Original annotation:", passage.get("original_type"))

    print("\nInspection completed successfully!")


if __name__ == "__main__":
    main()
