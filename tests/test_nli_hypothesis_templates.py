
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

print("Loading NLI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cpu")
model.to(device)

id2label = {
    int(k): v.lower()
    for k, v in model.config.id2label.items()
}

# Hand-crafted diagnostic examples.
# These are only for testing hypothesis construction.

examples = [
    {
        "name": "Broken Bow population",
        "passage": (
            "As of the census of 2010, there were 3,559 people "
            "residing in the city of Broken Bow."
        ),
        "current": (
            "The answer to the question "
            "'What is the population of Broken Bow?' is '3,559'."
        ),
        "natural": (
            "Broken Bow had a population of 3,559 people."
        ),
        "expected": "entailment"
    },
    {
        "name": "Bobby Carpenter sport",
        "passage": (
            "Bobby Carpenter is a former American football linebacker. "
            "He was drafted by the Dallas Cowboys in 2006."
        ),
        "current": (
            "The answer to the question "
            "'What sport is Bobby Carpenter associated with?' "
            "is 'American football'."
        ),
        "natural": (
            "Bobby Carpenter is associated with American football."
        ),
        "expected": "entailment"
    },
    {
        "name": "Ryan Davis sport",
        "passage": (
            "Ryan Davis made his AFL debut for the West Coast Eagles "
            "in the 2008 season and played Australian rules football."
        ),
        "current": (
            "The answer to the question "
            "'What sport is Ryan Davis associated with?' "
            "is 'basketball'."
        ),
        "natural": (
            "Ryan Davis is associated with basketball."
        ),
        "expected": "contradiction"
    }
]


def predict(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    results = {
        id2label[i]: probabilities[i].item()
        for i in range(len(probabilities))
    }

    predicted = max(results, key=results.get)

    return predicted, results


for example in examples:

    print("\n" + "=" * 75)
    print(example["name"])
    print("=" * 75)

    print("Expected:", example["expected"])

    for template_name in ["current", "natural"]:

        hypothesis = example[template_name]

        predicted, probabilities = predict(
            example["passage"],
            hypothesis
        )

        print(f"\nTemplate: {template_name.upper()}")
        print("Hypothesis:", hypothesis)
        print("Predicted:", predicted)

        for label, score in probabilities.items():
            print(f"  {label}: {score:.4f}")
