
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)

MODEL_NAME = "distilbert-base-cased-distilled-squad"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading QA model...")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
model.eval()

question = "What is the population of Broken Bow?"

context = """
As of the census of 2010, there were 3,559 people,
1,575 households, and 909 families residing in the city.
"""

print("\nRunning inference...")

encoding = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation="only_second",
    max_length=384,
    return_offsets_mapping=True
)

# Identify tokens belonging to the passage, not the question.
sequence_ids = encoding.sequence_ids(0)
context_mask = torch.tensor(
    [sid == 1 for sid in sequence_ids],
    dtype=torch.bool
)

# Remove metadata before passing inputs to the model.
model_inputs = {
    key: value
    for key, value in encoding.items()
    if key != "offset_mapping"
}

with torch.inference_mode():
    outputs = model(**model_inputs)

start_logits = outputs.start_logits[0].clone()
end_logits = outputs.end_logits[0].clone()

# Restrict answer selection to passage tokens.
start_logits[~context_mask] = float("-inf")
end_logits[~context_mask] = float("-inf")

start_index = int(torch.argmax(start_logits))
end_index = int(torch.argmax(end_logits))

if end_index < start_index or end_index - start_index > 30:
    raise RuntimeError("No valid answer span was identified.")

input_ids = encoding["input_ids"][0]

answer = tokenizer.decode(
    input_ids[start_index:end_index + 1],
    skip_special_tokens=True
)

print("\nQUESTION:", question)
print("PREDICTED ANSWER:", answer)

print("\nQA model test completed!")


if __name__ == "__main__":
    pass
