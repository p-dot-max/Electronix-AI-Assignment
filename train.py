import argparse
import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
import random
import json

# Defaults
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/train.jsonl"
SAVE_DIR = "model"
BATCH_SIZE = 8
EPOCHS = 10
LR = 3e-5
MAX_LEN = 128
SEED = 42
NUM_LABELS = 2
GRAD_CLIP = 1.0
DEVICE = torch.device("cpu")  # Force CPU for assignment

def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class JsonlDataset(Dataset):
    def __init__(self, filepath, tokenizer, label2id, max_len=128):
        self.data = []
        with open(filepath, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.data.append(entry)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.label2id[self.data[idx]["label"]]
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def main(args):
    setup_seeds(SEED)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quantization_config = {"load_in_8bit": True} if os.getenv("QUANTIZE", "false") == "true" else None
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
        quantization_config=quantization_config
    ).to(DEVICE)

    # Dataset
    dataset = JsonlDataset(args.data, tokenizer, {"negative": 0, "positive": 1}, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(dataloader)
    )

    # Training loop
    start_time = time.time()
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed. Avg loss: {total_loss / len(dataloader):.4f}")
    
    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    end_time = time.time()
    print(f"Model and tokenizer saved to {SAVE_DIR}")
    print(f"Fine-tuning took {end_time - start_time:.2f} seconds")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=DATA_PATH, help='data.jsonl path')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='total training epochs')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)