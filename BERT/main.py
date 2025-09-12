import os
import re
import time
import logging
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from pymongo import MongoClient
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
)

BASE_DOMAIN_MODEL = "google-bert/bert-large-uncased"
FALLBACK_MODEL = "google-bert/bert-large-uncased"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

WHITELIST_DOMAINS = {
    "geeksforgeeks.org",
    "wikipedia.org",
    "stackoverflow.com",
    "github.com",
    "medium.com",
    "arxiv.org",
    "python.org",
    "pypi.org",
    "docs.python.org"
}

WHITELIST_FILE = "whitelist.txt"

CODE_DENSITY_THRESHOLD = 0.02
CONFIDENCE_THRESHOLD = 0.70
ENTITY_SAFEGUARD_COUNT = 1

SCAN_SLEEP_SECONDS = 5

STAGE1_CSV = "darkweb_data_stage1_augmented.csv"
STAGE2_CSV = "darkweb_data_stage2_augmented.csv"

STAGE1_MODEL_DIR = "secbert_threat_model"
STAGE2_MODEL_DIR = "secbert_severity_model"

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

client = MongoClient("mongodb://localhost:27017/")
db = client["local"]
collection = db["test_1"]

severity_levels = ["Low", "Medium", "High"]
severity_weights = {"None": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}

regex_patterns = {
    "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "ip_addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "crypto_wallets": r"\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{25,39})\b",
    "usernames": r"\b@[a-zA-Z0-9_]{3,20}\b"
}

def load_whitelist():
    wl = set(WHITELIST_DOMAINS)
    if os.path.exists(WHITELIST_FILE):
        try:
            with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    dom = line.strip()
                    if dom:
                        wl.add(dom.lower())
        except Exception as e:
            logging.warning("Could not read whitelist file: %s", e)
    return wl

def clean_text(text):
    if text is None:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text()
    cleaned = re.sub(r"[^A-Za-z0-9\s\.;:{}()\[\]<>_`#\/\\\-\+@]", " ", cleaned)  # keep code & @ for usernames
    return re.sub(r"\s+", " ", cleaned).strip()

def read_file(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning("Error reading %s: %s", path, e)
            return None
    logging.warning("Could not decode %s with any encoding.", path)
    return None

def extract_entities(text):
    return {
        "emails": re.findall(regex_patterns["emails"], text),
        "ip_addresses": re.findall(regex_patterns["ip_addresses"], text),
        "crypto_wallets": re.findall(regex_patterns["crypto_wallets"], text),
        "usernames": re.findall(regex_patterns["usernames"], text),
    }

def keyword_density(text):
    words = text.lower().split()
    return len(words) / max(len(words), 1)

def compute_novelty(text, seen_hashes):
    content_hash = hash(text)
    if content_hash in seen_hashes:
        return 0.7
    else:
        seen_hashes.add(content_hash)
        return 1.5

def compute_score(severity, relevance, novelty, noise_factor):
    sev_weight = severity_weights.get(severity, 0.0)
    return round((relevance * sev_weight * novelty) / max(noise_factor, 0.1), 3)

def adjust_severity(severity, score):
    if severity == "High" and score < 1.0:
        return "Medium"
    if severity == "Medium" and score < 0.5:
        return "Low"
    return severity

def get_final_threat_level(severity, score):
    if severity == "High":
        return "Critical" if score >= 2.0 else "High (Low Confidence)"
    elif severity == "Medium":
        return "Medium" if score >= 1.5 else "Medium (Low Confidence)"
    else:
        return severity

class DarkWebDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[i], dtype=torch.long)
        }

def attempt_load_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logging.warning("Could not load tokenizer %s: %s", model_name, e)
        return None

def attempt_load_model(model_name, num_labels):
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    except Exception as e:
        logging.warning("Could not load model %s: %s", model_name, e)
        return None

def train_model(df, num_labels, save_dir, epochs=5, batch_size=4, base_model_name=BASE_DOMAIN_MODEL):
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Training DataFrame must contain 'text' and 'label' columns.")
    df = df.copy()
    df["text"] = df["text"].astype(str).apply(clean_text)

    tokenizer = attempt_load_tokenizer(base_model_name)
    fallback_used = False
    if tokenizer is None:
        logging.info("Falling back to %s tokenizer", FALLBACK_MODEL)
        tokenizer = attempt_load_tokenizer(FALLBACK_MODEL)
        fallback_used = True

    dataset = DarkWebDataset(df["text"].tolist(), df["label"].astype(int).tolist(), tokenizer)

    train_size = int(0.8 * len(dataset)) if len(dataset) > 1 else len(dataset)
    if train_size == 0:
        train_set = dataset
        val_set = []
    else:
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size) if len(val_set) > 0 else []

    model = attempt_load_model(base_model_name, num_labels=num_labels)
    if model is None:
        logging.info("Falling back to %s model", FALLBACK_MODEL)
        model = attempt_load_model(FALLBACK_MODEL, num_labels=num_labels)
        fallback_used = True

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_val_acc = 0
    patience = 1
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        if len(train_loader) == 0:
            logging.info("No training data for %s, skipping training loop.", save_dir)
            break
        for batch in tqdm(train_loader, desc=f"Training {save_dir} Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = 0.0
        if len(val_loader) > 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device)
                    )
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct += (preds.cpu() == batch["label"]).sum().item()
                    total += batch["label"].size(0)
            val_acc = correct / total if total > 0 else 0.0

        logging.info("Training %s Epoch %d | Avg loss: %.4f | Val acc: %.4f", save_dir, epoch+1, total_loss/max(1,len(train_loader)), val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logging.info("Saved best model to %s", save_dir)
        else:
            wait += 1
            if wait >= patience:
                logging.info("Early stopping triggered for %s", save_dir)
                break

def extract_domain_from_folder(folder_name):
    s = folder_name.lower().strip()
    s = re.sub(r"^https?://", "", s)
    s = s.split("/")[0]
    s = s.split(":")[0]
    if s.endswith(".onion") or "." not in s:
        return s
    parts = s.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return s

def code_density_score(text):
    code_tokens = [";", "{", "}", "def ", "class ", "import ", "printf(", "console.log", "<code>", "`", "=>", "function(", "void ", "int ", "#include"]
    count = 0
    for tk in code_tokens:
        count += text.count(tk)
    total_tokens = max(len(text.split()), 1)
    return count / total_tokens

def count_entities_in_text(text):
    ents = extract_entities(text)
    return len(ents["emails"]) + len(ents["ip_addresses"]) + len(ents["crypto_wallets"]) + len(ents["usernames"])

def scan_folders(base_dir, threat_model, severity_model, tokenizer_threat, tokenizer_severity, scanned, df_types, whitelist):
    results = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if folder_path in scanned or not os.path.isdir(folder_path):
            continue

        combined_text = ""
        seen_hashes = set()

        for file in os.listdir(folder_path):
            if not file.endswith(".html"):
                continue
            content = read_file(os.path.join(folder_path, file))
            if not content:
                continue
            cleaned = clean_text(content)
            combined_text += cleaned + " "

            # Save cleaned content into text files
            save_dir = os.path.join("content", folder)
            os.makedirs(save_dir, exist_ok=True)
            txt_file_path = os.path.join(save_dir, os.path.splitext(file)[0] + ".txt")
            try:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
            except Exception as e:
                logging.error("Failed to write cleaned file %s: %s", txt_file_path, e)

        if not combined_text.strip():
            scanned.add(folder_path)
            continue

        domain = extract_domain_from_folder(folder)

        if domain in whitelist:
            logging.info("Skipping %s (whitelisted domain: %s)", folder, domain)
            scanned.add(folder_path)
            continue

        # Precompute general heuristics for both stages
        cd_score = code_density_score(combined_text)
        ent_count = count_entities_in_text(combined_text)

        # ---------- Stage 1: Threat detection ----------
        is_suspicious = False
        threat_conf, threat_pred = 0.0, 0

        if threat_model is not None and tokenizer_threat is not None:
            inputs_threat = tokenizer_threat(
                combined_text, return_tensors="pt",
                truncation=True, padding="max_length", max_length=512
            ).to(device)
            with torch.no_grad():
                logits_threat = threat_model(**inputs_threat).logits
                probs_threat = F.softmax(logits_threat, dim=1)
                threat_pred = torch.argmax(probs_threat, dim=1).item()
                threat_conf = probs_threat[0][threat_pred].item()

            if threat_pred == 0:
                # Not a threat by model: keep if entities present (safety net)
                if ent_count >= ENTITY_SAFEGUARD_COUNT:
                    is_suspicious = True
                else:
                    scanned.add(folder_path)
                    logging.info("%s | No threat detected by model (conf=%0.3f)", folder, round(threat_conf, 3))
                    continue
            else:
                is_suspicious = True
        else:
            # Stage 2 path: assume stage 1 already filtered
            is_suspicious = True

        # ---------- Stage 2: Severity detection ----------
        sev_conf = 0.0
        highest_severity = "None"
        adjusted_severity = "Unknown"  # default when no severity model
        final_threat_level = "Potential" if is_suspicious else "None"
        downgrade_reason = None
        score = None

        if severity_model is not None and tokenizer_severity is not None:
            inputs_sev = tokenizer_severity(
                combined_text, return_tensors="pt",
                truncation=True, padding="max_length", max_length=512
            ).to(device)
            with torch.no_grad():
                logits_sev = severity_model(**inputs_sev).logits
                probs_sev = F.softmax(logits_sev, dim=1)
                sev_pred = torch.argmax(probs_sev, dim=1).item()
                sev_conf = probs_sev[0][sev_pred].item()
                highest_severity = severity_levels[sev_pred] if sev_pred < len(severity_levels) else "Unknown"

            if cd_score >= CODE_DENSITY_THRESHOLD and sev_conf < CONFIDENCE_THRESHOLD and ent_count < ENTITY_SAFEGUARD_COUNT:
                adjusted_severity = "None"
                final_threat_level = "Benign (Code-heavy, low confidence)"
                downgrade_reason = f"code_density={cd_score:.4f}, sev_conf={sev_conf:.3f}, entities={ent_count}"
                logging.info("Downgrading %s to None due to code-heavy heuristic (%s)", folder, downgrade_reason)
            else:
                adjusted_severity = adjust_severity(highest_severity, 0)
                if sev_conf < CONFIDENCE_THRESHOLD and ent_count < ENTITY_SAFEGUARD_COUNT:
                    if adjusted_severity == "High":
                        adjusted_severity = "Medium"
                    elif adjusted_severity == "Medium":
                        adjusted_severity = "Low"
                    downgrade_reason = f"low_confidence({sev_conf:.3f}), entities={ent_count}"
                    logging.info("Lowering severity for %s due to low confidence: %s", folder, downgrade_reason)

                if ent_count >= ENTITY_SAFEGUARD_COUNT:
                    adjusted_severity = highest_severity
                    downgrade_reason = None

                novelty = compute_novelty(combined_text, seen_hashes)
                density = keyword_density(combined_text)
                noise_factor = 1 + max(0, len(combined_text.split()) / 1000 - density)
                relevance = max(0.4, 0.3 * density + 0.7 * sev_conf)
                score = compute_score(adjusted_severity, relevance, novelty, noise_factor)
                adjusted_severity = adjust_severity(adjusted_severity, score)
                final_threat_level = get_final_threat_level(adjusted_severity, score)

        # Detect type keywords, if provided
        detected_type = "unknown"
        for t in df_types:
            try:
                if pd.isna(t):
                    continue
            except Exception:
                pass
            if str(t).lower() in combined_text.lower():
                detected_type = t
                break

        # If we ran severity and downgraded to None, skip; else keep (Stage 1 keeps Potential)
        if (severity_model is not None and tokenizer_severity is not None) and adjusted_severity == "None":
            scanned.add(folder_path)
            logging.info("%s | Final: None (skipped insertion) | Reason: %s", folder, downgrade_reason or "heuristic/whitelist")
            continue

        result_doc = {
            "onionsite_url": folder,
            "domain": domain,
            "severity": adjusted_severity,
            "final_threat_level": final_threat_level,
            "model_confidence": round(sev_conf, 3),
            "threat_detection_confidence": round(threat_conf, 3),
            "violation": detected_type,
            "score": round(score, 3) if score is not None else None,
            "entities": extract_entities(combined_text),
            "code_density": round(cd_score, 4),
            "downgrade_reason": downgrade_reason,
            "timestamp": datetime.utcnow(),
            "content_hash": hash(combined_text),
        }

        results.append(result_doc)
        scanned.add(folder_path)
        logging.info(
            "%s | Sev:%s | Threat:%s | SevConf:%0.3f | ThreatConf:%0.3f | Type:%s | Score:%s",
            folder, adjusted_severity, final_threat_level, round(sev_conf, 3), round(threat_conf, 3), detected_type, result_doc.get('score')
        )

    return results

def train_stage1():
    df = pd.read_csv(STAGE1_CSV)
    train_model(df, num_labels=2, save_dir=STAGE1_MODEL_DIR)

def train_stage2():
    df = pd.read_csv(STAGE2_CSV)
    train_model(df, num_labels=3, save_dir=STAGE2_MODEL_DIR)

def run_stage1(whitelist):
    tokenizer_threat = AutoTokenizer.from_pretrained(STAGE1_MODEL_DIR)
    model_threat = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL_DIR).to(device)
    model_threat.eval()

    # Stage 1: only threat model
    results = scan_folders(
        base_dir="scan",
        threat_model=model_threat,
        severity_model=None,      # not used in stage 1
        tokenizer_threat=tokenizer_threat,
        tokenizer_severity=None,  # not used in stage 1
        scanned=set(),
        df_types=[],
        whitelist=whitelist
    )
    return results

def run_stage2(stage1_results):
    tokenizer_sev = AutoTokenizer.from_pretrained(STAGE2_MODEL_DIR)
    model_sev = AutoModelForSequenceClassification.from_pretrained(STAGE2_MODEL_DIR).to(device)
    model_sev.eval()

    scanned = set()
    whitelist = load_whitelist()
    # If your stage2 CSV has labels describing violation types, we use them
    df_types = pd.read_csv(STAGE2_CSV)["label"].unique().tolist()

    # Stage 2: only severity model (assume folders are already filtered by stage 1)
    results = scan_folders(
        base_dir="scan",
        threat_model=None,               # already filtered in stage 1
        severity_model=model_sev,
        tokenizer_threat=None,
        tokenizer_severity=tokenizer_sev,
        scanned=scanned,
        df_types=df_types,
        whitelist=whitelist
    )
    return results

def main():
    whitelist = load_whitelist()

    if not os.path.exists(STAGE1_CSV):
        raise FileNotFoundError(f"Stage1 CSV not found: {STAGE1_CSV}. Create or place it in project root.")

    if not os.path.exists(STAGE1_MODEL_DIR):
        logging.info("Stage1 model not found. Training Stage1...")
        train_stage1()
    else:
        logging.info("Stage1 model found. Skipping training.")

    if not os.path.exists(STAGE2_CSV):
        raise FileNotFoundError(f"Stage2 CSV not found: {STAGE2_CSV}. Create or place it in project root.")

    if not os.path.exists(STAGE2_MODEL_DIR):
        logging.info("Stage2 model not found. Training Stage2...")
        train_stage2()
    else:
        logging.info("Stage2 model found. Skipping training.")

    logging.info("Running Stage1 inference...")
    stage1_results = run_stage1(whitelist)

    logging.info("Running Stage2 inference...")
    stage2_results = run_stage2(stage1_results)

    logging.info("Pipeline completed successfully.")
    return stage2_results

if __name__ == "__main__":
    results = main()
    print("Final Results:", results)
