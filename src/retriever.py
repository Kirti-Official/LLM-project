
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ramdocs_test.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ramdocs_retrieved.jsonl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3


def load_records():

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


def retrieve_passages(model, record):

    question = record["question"]
    documents = record["documents"]

    if not documents:
        return []

    passages = [
        document["text"]
        for document in documents
    ]

    # Encode question and passages
    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    passage_embeddings = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=16
    )

    # Calculate cosine similarity
    similarities = util.cos_sim(
        question_embedding,
        passage_embeddings
    )[0]

    # Select Top-K passages
    k = min(TOP_K, len(passages))

    top_indices = torch.topk(
        similarities,
        k=k
    ).indices.tolist()

    retrieved = []

    for rank, index in enumerate(top_indices, start=1):

        retrieved.append({
            "passage_id": f"P{index + 1}",
            "rank": rank,
            "text": passages[index],
            "similarity_score": round(
                similarities[index].item(), 4
            ),

            # Retained only for evaluation and auditing.
            # Not used to calculate similarity.
            "original_type": documents[index]["type"]
        })

    return retrieved


def main():

    print("=" * 60)
    print("VERDICT - Dense Passage Retrieval")
    print("=" * 60)

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Embedding model loaded successfully!")

    records = load_records()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_retrieved = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for index, record in enumerate(records):

            retrieved = retrieve_passages(
                model,
                record
            )

            result = {
                "sample_id": f"ramdocs_{index + 1:04d}",
                "question": record["question"],
                "retrieved_passages": retrieved
            }

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            total_retrieved += len(retrieved)

            if (index + 1) % 50 == 0:
                print(
                    f"Processed {index + 1}/{len(records)} records"
                )

    print("\nRETRIEVAL COMPLETED")
    print("Total questions:", len(records))
    print("Total retrieved passages:", total_retrieved)
    print("Top-K:", TOP_K)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
