from src.evidence_classifier import build_hypothesis

examples = [
    ("What is the population of Broken Bow?", "3,559"),
    ("What sport is Bobby Carpenter associated with?", "American football"),
    ("When was General Bryan born?", "February 8, 1900"),
    ("What is the length of the Moravica river?", "98 km"),
    ('Who are the directors of the film "Lahu Ke Do Rang"?',
     "Mahesh Bhatt")
]

for question, answer in examples:
    print("\nQuestion:", question)
    print("Candidate:", answer)
    print("Hypothesis:", build_hypothesis(question, answer))