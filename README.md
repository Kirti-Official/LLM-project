
# VERDICT
### Evidence-Aware Reasoning and Selective Abstention in Large Language Models Under Conflicting Contexts

## Overview

VERDICT is a research-oriented framework designed to improve the reliability of Large Language Models (LLMs) when answering questions using retrieved documents containing supporting, irrelevant, misleading, or conflicting information.

Instead of generating an answer for every question, VERDICT evaluates the available evidence and decides whether the system should answer or abstain.

## Key Objectives

- Retrieve relevant passages from external documents.
- Classify evidence as supporting, refuting, or irrelevant.
- Identify consistent, contradictory, ambiguous, and insufficient evidence.
- Make answer-or-abstain decisions based on evidence quality.
- Generate evidence-grounded responses with citations.

## Dataset

- Dataset: RAMDocs
- Source: HanNight/RAMDocs (Hugging Face)
- Split: Official test split
- Records used: 500

## Current System Pipeline

1. Dataset preparation
2. Dense passage retrieval
3. Passage screening
4. Candidate answer generation
5. Evidence classification using pretrained NLI
6. Evidence aggregation
7. Rule-based abstention decision
8. LLM response generation using Ollama

## Current Implementation

| Component | Implementation |
|---|---|
| Embedding model | all-MiniLM-L6-v2 |
| Retrieval | Sentence Transformers |
| Passage screening | Threshold-based filtering |
| Candidate QA | DistilBERT QA |
| Evidence classification | DeBERTa-v3-small NLI |
| Evidence aggregation | Rule-based |
| Abstention | Rule-based |
| Response generation | Llama 3.1 8B Instruct via Ollama |

## Experimental Results

Results obtained during the current development experiments:

- Retrieval Hit@3: 98.20%
- Passage screening relevant recall on exploratory holdout: 96.98%
- Candidate answer coverage: 98.60%
- Candidate answer token F1: 65.46%
- Evidence aggregation:
  - Consistent: 51.8%
  - Contradictory: 15.0%
  - Ambiguous: 13.0%
  - Insufficient: 20.2%
- Current rule-based abstention coverage: 5.8%

These are development-stage results, not final benchmark claims. The exploratory holdout is an internal split from the same official test dataset.

## Project Status

### Implemented
- RAMDocs preprocessing
- Dense retrieval
- Passage screening
- Candidate answer generation
- Pretrained NLI evidence classification
- Rule-based evidence aggregation
- Rule-based abstention
- Ollama response-generation integration
- Prompt-construction and unit tests

### Planned
- LoRA fine-tuning
- Learned evidence classification head
- Learned abstention head
- Confidence calibration
- Ablation studies
- Final benchmark evaluation

## Requirements

- Python 3.11.9
- Ollama
- Meta Llama 3.1 8B Instruct (`llama3.1:8b`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start Ollama and ensure the model is available:

```bash
ollama list
```

Run the response-generation pipeline:

```bash
python -m src.main
```

## Repository Structure

```text
VERDICT/
├── src/
├── tests/
├── scripts/
│   ├── analysis/
│   └── inspection/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

## Research Scope

VERDICT currently uses pretrained evidence classification and rule-based aggregation and abstention. Learned evidence and abstention heads, fine-tuning, and calibration are planned extensions and are not yet claimed as completed.

## Disclaimer

This project is an academic research prototype. Its outputs should not be treated as guaranteed factual conclusions.

## Dataset Attribution and License

This project uses the RAMDocs dataset.

- Dataset: HanNight/RAMDocs
- Source: https://huggingface.co/datasets/HanNight/RAMDocs
- Declared license: MIT

Please refer to the original dataset repository for its licensing information, provenance, acknowledgements, and usage considerations.

### Dataset Citation

Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal.
"Retrieval-Augmented Generation with Conflicting Evidence."
2025. arXiv:2504.13079.
