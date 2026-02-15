import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "deberta_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

package = {
    "tokenizer": tokenizer,
    "model": model
}

with open("model/deberta_fake_news.pkl", "wb") as f:
    pickle.dump(package, f)

print("Saved deberta_fake_news.pkl successfully!")
