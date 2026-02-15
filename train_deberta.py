import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DebertaV2Tokenizer

MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = DebertaV2Tokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

# 1️⃣ Load your dataset
df = pd.read_csv("data/train.csv")   # must contain: text, label (0/1)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)



MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = DebertaV2Tokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)



def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )



# 3️⃣ Convert to HuggingFace dataset
train_ds = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
val_ds = Dataset.from_dict({"text": list(val_texts), "label": list(val_labels)})

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

# 4️⃣ Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

args = TrainingArguments(
    output_dir="./deberta_model",
    eval_strategy="epoch",        # ← NEW NAME
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=50,
    load_best_model_at_end=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

# 6️⃣ Save trained model
model.save_pretrained("./deberta_model")
tokenizer.save_pretrained("./deberta_model")

print("DeBERTa training complete!")
