import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import requests
import PyPDF2
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import transformers

transformers.logging.set_verbosity_error()

start_total = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

nltk.download('wordnet')

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
data_path = os.path.join(desktop_path, "test_summarization")
articoli_path = os.path.join(data_path, "articoli")
abstracts_path = os.path.join(data_path, "Abstract.txt")
csv_file = os.path.join(desktop_path, "Summarization_Results.csv")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

abstracts = {}
with open(abstracts_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    pdf_name, abstract_text = None, ""
    for line in lines:
        if line.startswith("PDF "):
            if pdf_name and abstract_text:
                abstracts[pdf_name] = abstract_text.strip()
            pdf_name = line.split(": ")[-1].strip()
            abstract_text = ""
        elif line.startswith("Abstract:"):
            continue
        else:
            abstract_text += " " + line.strip()
    if pdf_name and abstract_text:
        abstracts[pdf_name] = abstract_text.strip()

def read_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f" Error reading {pdf_path}: {e}")
    return text

print("Loading PDF articles from:", articoli_path)
pdf_files = [f for f in os.listdir(articoli_path) if f.lower().endswith(".pdf")]
if not pdf_files:
    print("No PDF files found in the specified directory.")
    exit()
num_papers = min(1, len(pdf_files))
print(f"Found {len(pdf_files)} PDF files. Processing {num_papers} papers.")

def load_model(model_name, tokenizer_name, model_type="bart"):
    print(f"Loading {model_name}...")
    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    print(f"{model_name} loaded successfully.")
    return model, tokenizer

print("Loading Models...")
bart_model, bart_tokenizer = load_model("facebook/bart-large-cnn", "facebook/bart-large-cnn", model_type="bart")
t5_model, t5_tokenizer = load_model("t5-large", "t5-large", model_type="t5")
print("Models Loaded.")

def summarize(text, model, tokenizer, model_name, max_new_tokens=200):
    """Generates a summary using a local model"""
    if not text.strip():
        print(f"Skipping empty input for {model_name}.")
        return "Empty input", 0.0

    input_ids = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)
    print(f"Generating summary with {model_name}...")
    start_time = time.time()
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    print(f" {model_name} summary generated in {elapsed_time:.2f} seconds.")
    return summary, elapsed_time

def summarize_with_ollama(text, max_length=200):
    """Generates a summary using LLaMA 3 via Ollama API"""
    data = {
        "model": "llama3:latest",
        "prompt": f"Summarize the following text in {max_length} words:\n{text}",
        "stream": False
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f" [OLLAMA] Sending request to Ollama (Attempt {attempt}/{max_retries})...")
        start_time = time.time()
        try:
            response = requests.post(OLLAMA_URL, json=data, timeout=60)
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                print(f" LLaMA 3 summary generated in {elapsed_time:.2f} seconds.")
                return response.json().get("response", "Error parsing response").strip(), elapsed_time
            else:
                print(f"[OLLAMA] Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[OLLAMA] Connection failed: {e}")
    print(f"[OLLAMA] Summary generation failed after {max_retries} attempts.")
    return "Error with Ollama", 0.0

def summarize_with_deepseek(text, max_length=200):
    """Generates a summary using Deepseek via Ollama API"""
    data = {
        "model": "deepseek-r1:8b",  
        "prompt": f"Summarize the following text in {max_length} words: {text}",
        "stream": False
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"[DEEPSEEK] Sending request to Deepseek (Attempt {attempt}/{max_retries})...")
        start_time = time.time()
        try:
            response = requests.post(OLLAMA_URL, json=data, timeout=60)
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                print(f"Deepseek summary generated in {elapsed_time:.2f} seconds.")
                return response.json().get("response", "Error parsing response").strip(), elapsed_time
            elif response.status_code == 404:
                print("[DEEPSEEK] Error 404: The template 'deepseek-r1:8b' was not found. Make sure you downloaded it correctly.")
                break
            else:
                print(f"[DEEPSEEK] Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"[DEEPSEEK] Connection failed: {e}")
    print(f" [DEEPSEEK] Summary generation failed after {max_retries} attempts.")
    return "Error with Deepseek", 0.0

results = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for i in range(num_papers):
    file_name = pdf_files[i]
    pdf_path = os.path.join(articoli_path, file_name)
    full_paper = read_pdf(pdf_path).strip()
    reference_summary = abstracts.get(file_name, "").strip()
    
    if not full_paper:
        print(f"No text extracted from {file_name}. Skipping.")
        continue
    if not reference_summary:
        print(f"No reference abstract found for {file_name}. Skipping.")
        continue

    print(f"\n Processing Paper {i+1}/{num_papers}: {file_name}...")
    bart_summary, bart_time = summarize(full_paper, bart_model, bart_tokenizer, "BART")
    t5_summary, t5_time = summarize(full_paper, t5_model, t5_tokenizer, "T5")
    llama_summary, llama_time = summarize_with_ollama(full_paper)
    deepseek_summary, deepseek_time = summarize_with_deepseek(full_paper)

    for model, summary, time_taken in [
        ("BART", bart_summary, bart_time),
        ("T5", t5_summary, t5_time),
        ("LLaMA 3", llama_summary, llama_time),
        ("Deepseek", deepseek_summary, deepseek_time),
    ]:
        rouge_scores = scorer.score(reference_summary, summary)
        meteor = meteor_score([reference_summary.split()], summary.split())
        _, _, bert_f1 = bert_score([summary], [reference_summary], lang="en", rescale_with_baseline=False)
        results.append({
            "Model": model,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-2": rouge_scores["rouge2"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "METEOR": meteor,
            "BERTScore": bert_f1.mean().item(),
            "Time Taken (sec)": time_taken,
        })

df_results = pd.DataFrame(results)
df_avg = df_results.groupby("Model").mean()  

plt.figure(figsize=(12, 6))
df_avg.drop(columns=["Time Taken (sec)"]).plot(kind="bar", figsize=(12, 6))
plt.title("Summarization Performance: BART vs. T5 vs. LLaMA 3 vs. Deepseek (PDF Articles)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(6, 4))
df_avg["Time Taken (sec)"].plot(kind="bar")
plt.title("Execution Time Comparison (BART vs. T5 vs. LLaMA 3 vs. Deepseek) on PDF Articles")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

print("\n📊 Summarization Performance Table:")
print(df_avg)

total_time = time.time() - start_total
print(f"\n⏱ Total execution time: {total_time:.2f} seconds.")
