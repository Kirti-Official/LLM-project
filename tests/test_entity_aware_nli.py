
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

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

    return {
        id2label[i]: probs[i].item()
        for i in range(len(probs))
    }


tests = [
    {
        "name": "Broken Bow - Original sentence",
        "premise": (
            "As of the census of 2010, there were 3,559 people, "
            "1,575 households, and 909 families residing in the city."
        ),
        "hypothesis": (
            "The population of Broken Bow is 3,559."
        )
    },
    {
        "name": "Broken Bow - Generic hypothesis",
        "premise": (
            "As of the census of 2010, there were 3,559 people, "
            "1,575 households, and 909 families residing in the city."
        ),
        "hypothesis": (
            "The city had 3,559 people."
        )
    },
    {
        "name": "Broken Bow - Entity-resolved premise",
        "premise": (
            "Broken Bow had 3,559 people according to the 2010 census."
        ),
        "hypothesis": (
            "The population of Broken Bow is 3,559."
        )
    },
    {
        "name": "Lahu - Wrong organization candidate",
        "premise": (
            "Lahu Ke Do Rang is a 1979 Hindi movie directed by Mahesh Bhatt."
        ),
        "hypothesis": (
            "The director of Lahu Ke Do Rang is Raj Kapoor Films."
        )
    },
    {
        "name": "Lahu - Correct person candidate",
        "premise": (
            "Lahu Ke Do Rang is a 1979 Hindi movie directed by Mahesh Bhatt."
        ),
        "hypothesis": (
            "The director of Lahu Ke Do Rang is Mahesh Bhatt."
        )
    }
]


for test in tests:

    print("\n" + "=" * 75)
    print(test["name"])
    print("=" * 75)

    print("Premise:", test["premise"])
    print("Hypothesis:", test["hypothesis"])

    results = classify(
        test["premise"],
        test["hypothesis"]
    )

    predicted = max(results, key=results.get)

    print("\nPredicted:", predicted)

    for label, score in results.items():
        print(f"{label}: {score:.4f}")
