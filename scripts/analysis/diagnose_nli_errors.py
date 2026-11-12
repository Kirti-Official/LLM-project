
import json
import re
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.evidence_classifier import build_hypothesis

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

INPUT_FILE = Path(
    "data/processed/ramdocs_evidence_predictions.jsonl"
)

print("Loading NLI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = {
    int(k): v.lower()
    for k, v in model.config.id2label.items()
}


def classify(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    result = {
        id2label[i]: probs[i].item()
        for i in range(len(probs))
    }

    return result


records = []

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for line in file:
        records.append(json.loads(line))


diagnostic_questions = [
    "What is the population of Broken Bow?",
    'Who are the directors of the film "Lahu Ke Do Rang"?'
]

for question in diagnostic_questions:

    record = next(
        (r for r in records if r["question"] == question),
        None
    )

    if record is None:
        print("Question not found:", question)
        continue

    candidate = record.get("candidate_answer")

    hypothesis = build_hypothesis(question, candidate)

    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("CANDIDATE:", candidate)
    print("HYPOTHESIS:", hypothesis)
    print("=" * 80)

    for passage in record.get("retained_passages", []):

        passage_id = passage.get("passage_id")
        passage_text = passage.get("text", "")

        if not passage_text:
            continue

        print(f"\n--- {passage_id}: FULL PASSAGE ---")

        full_result = classify(
            passage_text,
            hypothesis
        )

        for label, score in full_result.items():
            print(f"{label}: {score:.4f}")

        # Extract sentences containing potentially relevant terms.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            passage_text
        )

        if question == diagnostic_questions[0]:
            keywords = ["3,559", "population", "people"]
        else:
            keywords = [
                "directed by",
                "director",
                "directed",
                "Mahesh Bhatt",
                "Raj Kapoor"
            ]

        relevant_sentences = [
            sentence for sentence in sentences
            if any(
                keyword.lower() in sentence.lower()
                for keyword in keywords
            )
        ]

        for sentence in relevant_sentences[:3]:

            print("\nRelevant sentence:")
            print(sentence[:500])

            sentence_result = classify(
                sentence,
                hypothesis
            )

            print("Sentence-level probabilities:")

            for label, score in sentence_result.items():
                print(f"  {label}: {score:.4f}")
