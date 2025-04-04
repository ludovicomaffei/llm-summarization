import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import requests
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import transformers
from datasets import load_dataset
import aiohttp

transformers.logging.set_verbosity_error()

start_total = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using Device: {device}")

nltk.download('wordnet')


print("🔄 Loading arXiv dataset...")
dataset = load_dataset(
        "scientific_papers", "arxiv", trust_remote_code=True, download_mode = "reuse_cache_if_exists",
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=7200)}}
)
num_papers = 250  
print(f"Loaded {num_papers} articles from arXiv dataset.")


desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
csv_file = os.path.join(desktop_path, "Summarization_Results.csv")
output_file = os.path.join(desktop_path, "summaries.txt")


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
    print(f"{model_name} summary generated in {elapsed_time:.2f} seconds.")
    return summary, elapsed_time


def summarize_with_ollama(text, max_length=200):
    """Generates a summary using LLaMA 3 via Ollama API"""
    data = {
        "model": "llama3:latest",
        "prompt": f"Summarize the following text in {max_length} words: {text}",
        "stream": False
    }
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        print(f"[OLLAMA] Sending request to Ollama (Attempt {attempt}/{max_retries})...")
        start_time = time.time()
        try:
            response = requests.post("http://127.0.0.1:11434/api/generate", json=data, timeout=60)
            if response.status_code == 200:
                elapsed_time = time.time() - start_time
                print(f"LLaMA 3 summary generated in {elapsed_time:.2f} seconds.")
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
            response = requests.post("http://127.0.0.1:11434/api/generate", json=data, timeout=60)
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
    print(f"[DEEPSEEK] Summary generation failed after {max_retries} attempts.")
    return "Error with Deepseek", 0.0


results = []
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


model_entries = {"BART": [], "T5": [], "LLaMA 3": [], "Deepseek": []}

for i in range(num_papers):
    
    full_paper = dataset[i]['article'].strip()
    reference_summary = dataset[i]['highlights'].strip()
    
    if not full_paper:
        print(f"Item {i+1} empty. Skipping.")
        continue
    if not reference_summary:
        print(f"No highlight for item {i+1}. Skipping.")
        continue

    print(f"\n Processing Article {i+1}/{num_papers}...")
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
        bert_value = bert_f1.mean().item()
        results.append({
            "Model": model,
            "ROUGE-1": rouge_scores["rouge1"].fmeasure,
            "ROUGE-2": rouge_scores["rouge2"].fmeasure,
            "ROUGE-L": rouge_scores["rougeL"].fmeasure,
            "METEOR": meteor,
            "BERTScore": bert_value,
            "Time Taken (sec)": time_taken,
        })
        
        model_entries[model].append({
            "index": i,
            "article": full_paper,
            "reference": reference_summary,
            "summary": summary,
            "bert": bert_value,
        })

df_results = pd.DataFrame(results)

df_avg = df_results.groupby("Model").mean()


plt.figure(figsize=(12, 6))
df_avg.drop(columns=["Time Taken (sec)"]).plot(kind="bar", figsize=(12, 6))
plt.title("Summarization Performance")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
plt.savefig(os.path.join(desktop_path, "performance_comparison.pdf"), format="pdf", bbox_inches="tight")
plt.close()


plt.figure(figsize=(6, 4))
df_avg["Time Taken (sec)"].plot(kind="bar")
plt.title("Execution Time Comparison")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
plt.savefig(os.path.join(desktop_path, "execution_time_comparison.pdf"), format="pdf", bbox_inches="tight")
plt.close()


print("\nSummarization Performance Table:")
print(df_avg)


total_time = time.time() - start_total
print(f"\n⏱ Total execution time: {total_time:.2f} seconds.")


with open(output_file, "w", encoding="utf-8") as f:
    for model in model_entries:
        entries = model_entries[model]
        if not entries:
            continue
        best = max(entries, key=lambda x: x["bert"])
        worst = min(entries, key=lambda x: x["bert"])
        f.write(f"Model: {model}\n")
        f.write("=== Highest BERTScore Summary ===\n")
        f.write(f"BERTScore: {best['bert']}\n")
        f.write("Article:\n")
        f.write(best['article'] + "\n")
        f.write("Summary:\n")
        f.write(best['summary'] + "\n\n")
        f.write("=== Lowest BERTScore Summary ===\n")
        f.write(f"BERTScore: {worst['bert']}\n")
        f.write("Article:\n")
        f.write(worst['article'] + "\n")
        f.write("Summary:\n")
        f.write(worst['summary'] + "\n")
        f.write("\n" + "="*50 + "\n\n")

print(f"\n Generated output file: {output_file}")
