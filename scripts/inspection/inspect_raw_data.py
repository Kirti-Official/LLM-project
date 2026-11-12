
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "ramdocs_test.jsonl"
)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        first_record = json.loads(next(
            line for line in file if line.strip()
        ))

    print("=" * 65)
    print("VERDICT - Original RAMDocs Data Inspection")
    print("=" * 65)

    print("\nAVAILABLE FIELDS")
    for key in first_record.keys():
        print("-", key)

    print("\nQUESTION")
    print(first_record.get("question"))

    print("\nGOLD ANSWERS")
    print(first_record.get("gold_answers"))

    print("\nWRONG ANSWERS")
    print(first_record.get("wrong_answers"))

    print("\nDOCUMENTS")

    documents = first_record.get("documents", [])

    for index, document in enumerate(documents[:3], start=1):
        print(f"\nDocument {index}")

        if isinstance(document, dict):
            for key, value in document.items():
                print(f"{key}: {str(value)[:500]}")
        else:
            print(str(document)[:500])

    print("\nInspection completed successfully!")


if __name__ == "__main__":
    main()
