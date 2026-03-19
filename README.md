# Text Summarization System — Fine-tuned Pegasus

Fine-tuned a seq2seq model on conversational data and deployed it via FastAPI for real-time summarization.

---

## Overview

This project fine-tunes the `pegasus-samsum` model on the SAMSum dataset — a corpus of messenger-style conversations with human-written summaries. The trained model is served via a FastAPI backend that accepts raw text and returns a clean summary.

Built with a production-style project structure: modular components, config-driven training, proper logging, and a clear separation between training and inference.

---

## Project Structure

```
TEXT-SUMMARIZER/
│
├── .github/                     # CI/CD configs (if any)
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
├── params.yaml                 # Hyperparameters
│
├── config/
│   ├── config.yaml             # Main config (paths, settings)
│   └── __init__.py
│
├── artifacts/                  # Generated during pipeline runs
│
├── logs/
│   └── continuous_logs.log
│
├── research/                   # Experimentation notebooks
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   ├── 4_model_evaluation.ipynb
│   └── textsummarizer.ipynb
│
├── src/
│   └── TextSummarizer/
│
│       ├── __init__.py
│
│       ├── components/         # Core ML logic
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   ├── model_evaluation.py
│       │   └── __init__.py
│
│       ├── pipeline/           # Orchestration layer
│       │   ├── prediction_pipeline.py
│       │   ├── stage1_DATA_ING.py
│       │   ├── stage2_data_transformation.py
│       │   ├── stage3_model_trainer.py
│       │   ├── stage4_model_evaluation.py
│       │   └── __init__.py
│
│       ├── config/             # Config handling (code)
│       │   ├── configuration.py
│       │   └── __init__.py
│
│       ├── constants/          # Static values
│       │   └── __init__.py
│
│       ├── entity/             # Data classes / schemas
│       │   └── __init__.py
│
│       ├── utils/              # Helper functions
│       │   ├── common.py
│       │   └── __init__.py
│
│       ├── logging/            # Logging setup
│       │   └── __init__.py
│
├── app.py                      # FastAPI app (serving)
├── main.py                     # Pipeline runner / entry point
├── template.py                 # Project scaffolding
```

---

## Model

| Detail | Value |
|--------|-------|
| Base model | `google/pegasus-samsum` (via `transformersbook`) |
| Dataset | SAMSum (16k conversation-summary pairs) |
| Task | Abstractive text summarization (seq2seq) |
| Evaluation | ROUGE-1, ROUGE-2, ROUGE-L |

---

## Training Setup

- HuggingFace `Trainer` API with `Seq2SeqTrainer`
- Mixed precision training (`fp16=True`)
- Gradient accumulation for memory efficiency
- Custom preprocessing: tokenization, truncation, and decoder label alignment for seq2seq
- Config-driven hyperparameters (batch size, epochs, LR, etc.)

---

## API

Once running, the FastAPI server exposes a single endpoint:

**`POST /predict`**

```json
{
  "text": "Hannah: Hey, are you free tonight? ..."
}
```

**Response:**

```json
{
  "summary": "Hannah asks about plans for tonight."
}
```

---

## Setup & Usage

**1. Clone the repo**
```bash
git clone https://github.com/your-username/text-summarization-pegasus.git
cd text-summarization-pegasus
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the training pipeline**
```bash
python main.py
```

**4. Start the API server**
```bash
uvicorn app:app --reload
```

**5. Test the endpoint**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "your conversation here"}'
```

---

## Requirements

```
transformers
datasets
torch
evaluate
rouge_score
fastapi
uvicorn
accelerate
```

---

## Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | — |
| ROUGE-2 | — |
| ROUGE-L | — |

> Fill in your actual scores after training. Even rough numbers here make a difference.

---

## What I'd improve with more time

- Add a Streamlit front-end for non-technical users
- Experiment with `facebook/bart-large-cnn` as an alternative base model
- Add confidence scoring to the inference output
- Containerize with Docker for cleaner deployment

---

## References

- [SAMSum Dataset](https://huggingface.co/datasets/samsum)
- [Pegasus Paper](https://arxiv.org/abs/1912.08777)
- [HuggingFace Seq2Seq Docs](https://huggingface.co/docs/transformers/model_doc/pegasus)
