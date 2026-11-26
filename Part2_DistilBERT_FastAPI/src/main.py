from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

app = FastAPI(title="Customer Support Ticket Classifier")

BASE_DIR = Path(__file__).resolve().parent.parent

CATEGORY_MODEL_PATH = str(
    BASE_DIR / "results" / "distilbert_category_model"
)

PRIORITY_MODEL_PATH = str(
    BASE_DIR / "results" / "distilbert_priority_model"
)

# Load both models once at startup
category_tokenizer = AutoTokenizer.from_pretrained(
    CATEGORY_MODEL_PATH
)
category_model = AutoModelForSequenceClassification.from_pretrained(
    CATEGORY_MODEL_PATH
)

priority_tokenizer = AutoTokenizer.from_pretrained(
    PRIORITY_MODEL_PATH
)
priority_model = AutoModelForSequenceClassification.from_pretrained(
    PRIORITY_MODEL_PATH
)

category_model.eval()
priority_model.eval()


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1)


@app.get("/")
def home():
    return {
        "message": "Customer Support Ticket Classifier API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):

    # Category prediction
    category_inputs = category_tokenizer(
        ticket.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        category_outputs = category_model(
            **category_inputs
        )

    category_probabilities = torch.softmax(
        category_outputs.logits,
        dim=1
    )

    category_prediction = torch.argmax(
        category_probabilities,
        dim=1
    ).item()

    category = category_model.config.id2label[
        category_prediction
    ]

    # Priority prediction
    priority_inputs = priority_tokenizer(
        ticket.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        priority_outputs = priority_model(
            **priority_inputs
        )

    priority_probabilities = torch.softmax(
        priority_outputs.logits,
        dim=1
    )

    priority_prediction = torch.argmax(
        priority_probabilities,
        dim=1
    ).item()

    priority = priority_model.config.id2label[
        priority_prediction
    ]

    confidence = category_probabilities[
        0, category_prediction
    ].item()

    return {
        "category": category,
        "priority": priority,
        "confidence": round(confidence, 4)
    }