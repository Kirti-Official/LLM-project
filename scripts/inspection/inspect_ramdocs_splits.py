
from datasets import get_dataset_split_names, load_dataset

DATASET_NAME = "HanNight/RAMDocs"

def main():
    print("=" * 60)
    print("VERDICT - RAMDocs Dataset Split Inspection")
    print("=" * 60)

    splits = get_dataset_split_names(DATASET_NAME)

    print("\nAvailable splits:")
    for split in splits:
        print(f"- {split}")

    print("\nSplit sizes:")
    for split in splits:
        dataset = load_dataset(DATASET_NAME, split=split)
        print(f"{split}: {len(dataset)} records")

    print("\nInspection completed successfully!")

if __name__ == "__main__":
    main()
