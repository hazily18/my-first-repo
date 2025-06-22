from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gdown
import zipfile
import os

app = FastAPI()

MODEL_ZIP_URL = "https://drive.google.com/uc?id=1BN4_EHCcQ3zPVjAxUUgnpnLgKKXcDNx1"
MODEL_DIR = "model/final"
MODEL_ZIP_FILE = "model.zip"

# Automatically download and unzip model if not already present
if not os.path.exists(MODEL_DIR):
    print("Downloading model...")
    gdown.download(MODEL_ZIP_URL, MODEL_ZIP_FILE, quiet=False)
    print("Unzipping model...")
    with zipfile.ZipFile(MODEL_ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall("model")
    print("Model ready!")

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()
print("Model loaded.")


class CodeInput(BaseModel):
    python_code: str


@app.post("/translate")
def translate_code(data: CodeInput):
    inputs = tokenizer(data.python_code, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=1000)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"nodejs_code": translated}


@app.get("/")
def root():
    return {"message": "Python to Node.js translator is running!"}
