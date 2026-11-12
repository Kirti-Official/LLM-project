
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

print("=" * 65)
print("VERDICT - NLI Model Test")
print("=" * 65)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading NLI model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("\nModel label mapping:")
print(model.config.id2label)

examples = [
    {
        "premise": (
            "As of the 2010 census, Broken Bow had "
            "3,559 people residing in the city."
        ),
        "hypothesis": (
            "The population of Broken Bow was "
            "3,559 people in 2010."
        )
    },
    {
        "premise": (
            "As of the 2010 census, Broken Bow had "
            "3,559 people residing in the city."
        ),
        "hypothesis": (
            "The population of Broken Bow was "
            "10,000 people in 2010."
        )
    },
    {
        "premise": (
            "Broken Bow is located in Nebraska, "
            "approximately 10 miles southeast of its "
            "geographic center."
        ),
        "hypothesis": (
            "The population of Broken Bow was "
            "3,559 people in 2010."
        )
    }
]


def predict_nli(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_index = int(torch.argmax(probabilities))

    label_mapping = model.config.id2label

    predicted_label = label_mapping[predicted_index]

    return predicted_label, probabilities.tolist()


for index, example in enumerate(examples, start=1):

    label, probabilities = predict_nli(
        example["premise"],
        example["hypothesis"]
    )

    print(f"\nExample {index}")
    print("-" * 65)

    print("Premise:", example["premise"])
    print("Hypothesis:", example["hypothesis"])

    print("Predicted NLI label:", label)

    print("Class probabilities:")

    for class_index, probability in enumerate(probabilities):
        class_name = model.config.id2label[class_index]

        print(
            f"{class_name}: {probability:.4f}"
        )

print("\nNLI model test completed!")
