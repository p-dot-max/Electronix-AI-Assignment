import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class BinarySentimentModel:
    def __init__(self, model_dir="model", model_name="distilbert-base-uncased"):
        self.device = torch.device("cpu")
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        # Check if fine-tuned model exists by looking for config.json
        model_path = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        quantization_config = {"load_in_8bit": True} if os.getenv("QUANTIZE", "false") == "true" else None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            quantization_config=quantization_config
        ).to(self.device)
        self.model.eval()
        
        # Hot-reload model weights
        observer = Observer()
        observer.schedule(ModelReloadHandler(self), path=model_dir, recursive=False)
        observer.start()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score = probs[0][1].item()  # Probability of positive class
        label = "positive" if score > 0.5 else "negative"
        return {"label": label, "score": score}

class ModelReloadHandler(FileSystemEventHandler):
    def __init__(self, model_instance):
        self.model_instance = model_instance
    
    def on_modified(self, event):
        if event.src_path.endswith(".bin"):
            print("Reloading model weights...")
            self.model_instance.model = AutoModelForSequenceClassification.from_pretrained(
                "model", num_labels=2
            ).to(self.model_instance.device)
            print("Model reloaded")