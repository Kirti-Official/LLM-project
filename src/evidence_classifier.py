import re
import json
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_with_candidates.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "ramdocs_evidence_predictions.jsonl"
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

MAX_LENGTH = 512
PREMISE_WINDOW_SIZE = 350
WINDOW_OVERLAP = 100


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



def build_hypothesis(question, candidate_answer):

    q = question.strip().rstrip("?").strip()
    q_lower = q.lower()
    answer = candidate_answer.strip()

    # 1. Population questions
    match = re.match(
        r"what is the population of (.+)",
        q_lower
    )

    if match:
        return f"The city had {answer} people."

    # 2. Sport-related questions
    match = re.match(
        r"what sport is (.+) associated with",
        q,
        re.IGNORECASE
    )

    if match:
        subject = match.group(1).strip()
        return f"{subject} is associated with {answer}."

    # 3. Birth date questions
    match = re.match(
        r"when was (.+) born",
        q,
        re.IGNORECASE
    )

    if match:
        subject = match.group(1).strip()
        return f"{subject} was born on {answer}."

    # 4. Length questions
    match = re.match(
        r"what is the length of (.+)",
        q,
        re.IGNORECASE
    )

    if match:
        subject = match.group(1).strip()
        return f"The length of {subject} is {answer}."

    # 5. Film director questions
    match = re.match(
        r"who are the directors of the film (.+)",
        q,
        re.IGNORECASE
    )

    if match:
        subject = match.group(1).strip()
        return f"The directors of the film {subject} are {answer}."

    # 6. General fallback
    return f"The answer to the question '{q}' is '{answer}'."



def get_label_mapping(model):
    mapping = {}

    for index, label in model.config.id2label.items():
        normalized = label.lower()

        if normalized in {"entailment", "contradiction", "neutral"}:
            mapping[normalized] = index

    required = {"entailment", "contradiction", "neutral"}

    if set(mapping) != required:
        raise ValueError(
            f"Unexpected NLI labels: {model.config.id2label}"
        )

    return mapping


def split_into_windows(text, tokenizer):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not token_ids:
        return []

    step = PREMISE_WINDOW_SIZE - WINDOW_OVERLAP
    windows = []

    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + PREMISE_WINDOW_SIZE]

        if not chunk:
            break

        windows.append(
            tokenizer.decode(
                chunk,
                skip_special_tokens=True
            )
        )

        if start + PREMISE_WINDOW_SIZE >= len(token_ids):
            break

    return windows


def classify_passage(
    model,
    tokenizer,
    label_mapping,
    passage_text,
    hypothesis
):
    windows = split_into_windows(
        passage_text,
        tokenizer
    )

    if not windows:
        return None

    window_results = []

    for window in windows:

        inputs = tokenizer(
            window,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        result = {
            label: float(probabilities[index])
            for label, index in label_mapping.items()
        }

        window_results.append(result)

    # Keep the strongest observed class probability across windows.
    max_probabilities = {
        label: max(result[label] for result in window_results)
        for label in label_mapping
    }

    predicted_label = max(
        max_probabilities,
        key=max_probabilities.get
    )

    verdict_mapping = {
        "entailment": "supports",
        "contradiction": "refutes",
        "neutral": "irrelevant"
    }

    return {
        "predicted_label": verdict_mapping[predicted_label],
        "nli_label": predicted_label,
        "probabilities": max_probabilities,
        "windows_processed": len(windows)
    }


def main():
    print("=" * 70)
    print("VERDICT - Evidence Classification")
    print("=" * 70)

    records = load_records()

    print("\nQuestions:", len(records))
    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    label_mapping = get_label_mapping(model)

    print("\nModel label mapping:", label_mapping)

    total_passages = 0
    classified_passages = 0
    skipped_questions = 0

    label_counts = {
        "supports": 0,
        "refutes": 0,
        "irrelevant": 0
    }

    for record in tqdm(records):

        question = record["question"]
        candidate_answer = record.get("candidate_answer")

        record["evidence_predictions"] = []

        if not candidate_answer:
            record["evidence_status"] = "no_candidate_answer"
            skipped_questions += 1
            continue

        hypothesis = build_hypothesis(
            question,
            candidate_answer
        )

        record["candidate_hypothesis"] = hypothesis

        for passage in record.get("retained_passages", []):

            total_passages += 1

            result = classify_passage(
                model,
                tokenizer,
                label_mapping,
                passage.get("text", ""),
                hypothesis
            )

            if result is None:
                continue

            prediction = {
                "passage_id": passage.get("passage_id"),
                **result
            }

            record["evidence_predictions"].append(prediction)

            label_counts[result["predicted_label"]] += 1
            classified_passages += 1

        record["evidence_status"] = "classified"

    save_records(records)

    print("\n" + "=" * 70)
    print("EVIDENCE CLASSIFICATION RESULTS")
    print("=" * 70)

    print("Total questions:", len(records))
    print("Questions skipped:", skipped_questions)
    print("Total retained passages:", total_passages)
    print("Classified passages:", classified_passages)

    print("\nPREDICTION DISTRIBUTION")
    print("-" * 70)

    for label, count in label_counts.items():
        percentage = (
            count / classified_passages * 100
            if classified_passages else 0
        )

        print(
            f"{label.capitalize()}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\nOutput file:")
    print(OUTPUT_FILE)

    print("\nEvidence classification completed!")


if __name__ == "__main__":
    main()
