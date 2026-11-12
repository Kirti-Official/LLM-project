
import json
import re
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_screened_with_answers.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_with_candidates.jsonl"
)

MODEL_NAME = "distilbert-base-cased-distilled-squad"

MAX_LENGTH = 384
STRIDE = 128
MAX_ANSWER_TOKENS = 30


def load_records():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_records(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(answer):
    answer = re.sub(r"\s+", " ", answer).strip()

    # Fix numbers decoded with spaces after commas.
    answer = re.sub(r"(?<=\d),\s+(?=\d)", ",", answer)

    return answer


def extract_answer(model, tokenizer, question, context):

    encoding = tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = encoding["offset_mapping"]
    overflow_mapping = encoding["overflow_to_sample_mapping"]

    # Remove metadata before model inference.
    model_inputs = {
        key: value
        for key, value in encoding.items()
        if key not in {
            "offset_mapping",
            "overflow_to_sample_mapping"
        }
    }

    model.eval()

    with torch.inference_mode():
        outputs = model(**model_inputs)

    best_prediction = None

    for window_index in range(len(offsets)):

        start_logits = outputs.start_logits[window_index].clone()
        end_logits = outputs.end_logits[window_index].clone()

        sequence_ids = encoding.sequence_ids(window_index)

        context_mask = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            dtype=torch.bool
        )

        if not context_mask.any():
            continue

        start_logits[~context_mask] = float("-inf")
        end_logits[~context_mask] = float("-inf")

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        valid_positions = torch.where(context_mask)[0]

        top_k = min(20, len(valid_positions))

        top_starts = torch.topk(
            start_probs,
            top_k
        ).indices.tolist()

        top_ends = torch.topk(
            end_probs,
            top_k
        ).indices.tolist()

        for start_index in top_starts:

            for end_index in top_ends:

                if end_index < start_index:
                    continue

                if end_index - start_index + 1 > MAX_ANSWER_TOKENS:
                    continue

                start_offset = int(
                    offsets[window_index][start_index][0]
                )

                end_offset = int(
                    offsets[window_index][end_index][1]
                )

                if end_offset <= start_offset:
                    continue

                answer = context[start_offset:end_offset]
                answer = normalize_answer(answer)

                if not answer:
                    continue

                score = float(
                    torch.sqrt(
                        start_probs[start_index]
                        * end_probs[end_index]
                    )
                )

                if (
                    best_prediction is None
                    or score > best_prediction["score"]
                ):
                    best_prediction = {
                        "answer": answer,
                        "score": score
                    }

    return best_prediction


def generate_candidate(model, tokenizer, question, passages):

    predictions = []

    for passage in passages:

        context = passage.get("text", "").strip()

        if not context:
            continue

        prediction = extract_answer(
            model,
            tokenizer,
            question,
            context
        )

        if prediction:

            predictions.append({
                "answer": prediction["answer"],
                "score": prediction["score"],
                "passage_id": passage.get("passage_id")
            })

    if not predictions:
        return None

    return max(
        predictions,
        key=lambda item: item["score"]
    )


def main():

    print("=" * 65)
    print("VERDICT - Candidate Answer Generation")
    print("=" * 65)

    records = load_records()

    print("\nTotal questions:", len(records))
    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    successful = 0
    unsuccessful = 0

    print("\nGenerating candidate answers...\n")

    for record in tqdm(records):

        question = record["question"]

        passages = record.get("retained_passages", [])

        prediction = generate_candidate(
            model,
            tokenizer,
            question,
            passages
        )

        if prediction:

            record["candidate_answer"] = prediction["answer"]

            record["candidate_answer_metadata"] = {
                "model": MODEL_NAME,
                "method": "extractive_question_answering",
                "source_passage_id": prediction["passage_id"],
                "qa_score": prediction["score"]
            }

            record["candidate_answer_source"] = "qa_model"

            successful += 1

        else:

            record["candidate_answer"] = None
            record["candidate_answer_metadata"] = None
            record["candidate_answer_source"] = "not_generated"

            unsuccessful += 1

    save_records(records)

    print("\n" + "=" * 65)
    print("CANDIDATE GENERATION RESULTS")
    print("=" * 65)

    print("Total questions:", len(records))
    print("Answers generated:", successful)
    print("No answer generated:", unsuccessful)

    if records:
        print(
            "Generation rate:",
            f"{successful / len(records) * 100:.2f}%"
        )

    print("\nFIRST FIVE PREDICTIONS")
    print("-" * 65)

    for record in records[:5]:

        print("\nQuestion:", record["question"])
        print("Candidate:", record["candidate_answer"])

        metadata = record.get("candidate_answer_metadata")

        if metadata:
            print("Source passage:", metadata["source_passage_id"])
            print("QA score:", round(metadata["qa_score"], 4))

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nCandidate answer generation completed!")


if __name__ == "__main__":
    main()
