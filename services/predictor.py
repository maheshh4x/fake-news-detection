import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "maheshchandra07/fake-news-deberta"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()



def predict_news(text: str):
    if not text or len(text.strip()) == 0:
        return "No text provided", 0.0, 0.0, 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    real_prob = float(probs[0][0])
    fake_prob = float(probs[0][1])

    if fake_prob > real_prob:
        return "Fake News", fake_prob, real_prob, fake_prob
    else:
        return "Real News", real_prob, real_prob, fake_prob
