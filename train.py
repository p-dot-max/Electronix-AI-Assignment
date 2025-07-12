import argparse
import numpy as np
import os 
import sys
import time
import torch
import torch.nn as nn
import yaml
import torch.distributed as dist
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import random

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# https://github.com/AILab-CVC/SEED/blob/main/SEED_Tokenizer/train.py
def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# https://gist.github.com/Norod/7379927a41fc37a448ca5433beec0061
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
        '''
        {"text": "This movie wasn't that great", "label": "bad"}
        label2id = {"positive": 1, "negative": 0}
        '''
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def main(args):
    setup_seeds(SEED)

    # device = torch.device('cpu')
    model_name = MODEL_NAME
    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}

    token = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        name_label = 2,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    # Dataset
    dataset = JsonlDataset(args.data, token, label2id)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(dataloader)
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch + 1} completed. Avg loss: {total_loss / len(dataloader):.4f}")

    
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    token.save_pretrained("model")
    print("Model and tokenizer saved to ./Backend/model")
pass


# Parsers for the model training arguments
def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, required=True, help='data.jsonl path')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate scheduler')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def run(**kwargs):
    # Usage: import train; train.run(--data 'data.jsonl' --epochs 3 --lr 3e-5)
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



