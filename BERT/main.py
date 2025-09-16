import os
import re
import logging
import uuid
import json
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
from decimal import Decimal

import boto3
import botocore

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# ---------- Config ----------
BASE_DOMAIN_MODEL = "google-bert/bert-large-uncased"
FALLBACK_MODEL = "google-bert/bert-large-uncased"

STAGE1_CSV = "darkweb_data_stage1_augmented.csv"
STAGE2_CSV = "darkweb_data_stage2_augmented.csv"

STAGE1_MODEL_DIR = "secbert_threat_model"
STAGE2_MODEL_DIR = "secbert_severity_model"

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

# how long to wait between scan cycles (seconds)
SCAN_SLEEP_SECONDS = 5

regex_patterns = {
    "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "cves": r"\bCVE-\d{4}-\d{4,7}\b",
    "ip_addresses": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "crypto_wallets": r"\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{25,39})\b",
    "usernames": r"\b@[a-zA-Z0-9_]{3,20}\b"
}

severity_levels = ["Low", "Medium", "High"]
severity_weights = {"None": 0.0, "Low": 1.0, "Medium": 2.0, "High": 3.0}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# ---------- DynamoDB setup ----------
# Table name and region from your input
DDB_TABLE_NAME = "darkweb-threats"
DDB_REGION = "ap-south-1"

try:
    ddb_resource = boto3.resource("dynamodb", region_name=DDB_REGION)
    ddb_table = ddb_resource.Table(DDB_TABLE_NAME)
    # optionally check table exists (this will raise if not)
    _ = ddb_table.table_status  # will trigger metadata retrieval
    logging.info("Connected to DynamoDB table %s in %s", DDB_TABLE_NAME, DDB_REGION)
except botocore.exceptions.BotoCoreError as e:
    logging.error("Failed to initialize DynamoDB resource: %s", e)
    ddb_table = None

# ---------- Utilities ----------
def load_whitelist():
    wl = set(d.lower() for d in WHITELIST_DOMAINS)
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
    try:
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text()
    except Exception:
        cleaned = str(text)
    cleaned = re.sub(r"[^A-Za-z0-9\s\.;:{}()\[\]<>_`#\/\\\-\+@\-]", " ", cleaned)
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

def extract_entities_simple(text):
    """Return only entities we want to keep in schema: credentials (emails and some usernames) and CVEs."""
    emails = re.findall(regex_patterns["emails"], text)
    cves = re.findall(regex_patterns["cves"], text, flags=re.IGNORECASE)
    usernames = re.findall(regex_patterns["usernames"], text)
    # normalize/dedupe
    emails = list(dict.fromkeys([e.lower() for e in emails]))
    cves = list(dict.fromkeys([c.upper() for c in cves]))
    usernames = list(dict.fromkeys([u.lstrip("@") for u in usernames]))
    credentials = []
    for e in emails:
        credentials.append({
            "type": "email",
            "value": e,
            "confidence": 0.95,
            "match_type": "regex"
        })
    for u in usernames:
        credentials.append({
            "type": "username",
            "value": u,
            "confidence": 0.75,
            "match_type": "regex"
        })
    return {"credentials": credentials, "cves": cves}

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
    return round((relevance * sev_weight * novelty) / max(noise_factor, 0.1), 4)

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
    elif severity == "Low":
        return "Low"
    else:
        return "Potential"

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
    ents = extract_entities_simple(text)
    return len(ents["credentials"]) + len(ents["cves"])

def safe_normalize_risk(raw_score):
    """Normalize a raw score (arbitrary scale) into 0-100 integer."""
    if raw_score is None:
        return 0
    scaled = int(round(max(0.0, min(raw_score * 25.0, 100.0))))
    return scaled

def urgency_from_risk(risk_score):
    if risk_score >= 70:
        return "high"
    if risk_score >= 40:
        return "medium"
    return "low"

def action_from_urgency(urgency):
    if urgency == "high":
        return "notify-corp"
    if urgency == "medium":
        return "triage"
    return "monitor"

# ---------- Dataset + Training helpers ----------
class DarkWebDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

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

def train_model(df, num_labels, save_dir, epochs=5, batch_size=2, base_model_name=BASE_DOMAIN_MODEL):
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

# ---------- Scanning / Inference ----------
def scan_folders(base_dir,
                 threat_model,
                 severity_model,
                 tokenizer_threat,
                 tokenizer_severity,
                 scanned,
                 df_types,
                 whitelist,
                 source_map=None,
                 search_keyword=None):
    """
    Scans each unprocessed folder in base_dir and returns a list of simplified documents for newly processed folders.
    The 'scanned' set is mutated in-place so callers can keep it between cycles.
    """
    results = []
    if source_map is None:
        source_map = {}

    # create base_dir if not exists (so loop doesn't break)
    if not os.path.exists(base_dir):
        logging.warning("Scan base_dir %s does not exist. Creating it.", base_dir)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        # continue if not directory or already scanned
        if folder_path in scanned or not os.path.isdir(folder_path):
            continue

        combined_text = ""
        total_bytes = 0
        http_status = None
        fetch_ok = False

        for file in os.listdir(folder_path):
            if not file.endswith(".html"):
                continue
            content = read_file(os.path.join(folder_path, file))
            if not content:
                continue
            cleaned = clean_text(content)
            combined_text += cleaned + " "
            total_bytes += len(content.encode("utf-8", errors="ignore"))

        if not combined_text.strip():
            scanned.add(folder_path)
            logging.info("Skipping %s: no html content found.", folder_path)
            continue

        domain = extract_domain_from_folder(folder)

        if domain in whitelist:
            logging.info("Skipping %s (whitelisted domain: %s)", folder, domain)
            scanned.add(folder_path)
            continue

        cd_score = code_density_score(combined_text)
        ent_count = count_entities_in_text(combined_text)
        seen_hashes_local = set()

        threat_conf, threat_pred = 0.0, None
        if threat_model is not None and tokenizer_threat is not None:
            try:
                inputs_threat = tokenizer_threat(
                    combined_text, return_tensors="pt",
                    truncation=True, padding="max_length", max_length=512
                ).to(device)
                with torch.no_grad():
                    logits_threat = threat_model(**inputs_threat).logits
                    probs_threat = F.softmax(logits_threat, dim=1)
                    threat_pred = torch.argmax(probs_threat, dim=1).item()
                    threat_conf = float(probs_threat[0][threat_pred].item())
            except Exception as e:
                logging.warning("Threat model inference failed for %s: %s", folder, e)
                threat_conf, threat_pred = 0.0, None

        # Severity model
        sev_conf, sev_pred = 0.0, None
        highest_severity = "None"
        if severity_model is not None and tokenizer_severity is not None:
            try:
                inputs_sev = tokenizer_severity(
                    combined_text, return_tensors="pt",
                    truncation=True, padding="max_length", max_length=512
                ).to(device)
                with torch.no_grad():
                    logits_sev = severity_model(**inputs_sev).logits
                    probs_sev = F.softmax(logits_sev, dim=1)
                    sev_pred = torch.argmax(probs_sev, dim=1).item()
                    sev_conf = float(probs_sev[0][sev_pred].item())
                    if sev_pred < len(severity_levels):
                        highest_severity = severity_levels[sev_pred]
                    else:
                        highest_severity = "Unknown"
            except Exception as e:
                logging.warning("Severity model inference failed for %s: %s", folder, e)
                sev_conf, sev_pred = 0.0, None

        is_suspicious = False
        if threat_pred is not None:
            if isinstance(threat_pred, int) and threat_pred != 0:
                is_suspicious = True
            else:
                if ent_count >= ENTITY_SAFEGUARD_COUNT:
                    is_suspicious = True
        else:
            if highest_severity != "None" or ent_count >= ENTITY_SAFEGUARD_COUNT:
                is_suspicious = True

        adjusted_severity = highest_severity
        downgrade_reason = None
        score_val = None
        final_threat_level = "Potential" if is_suspicious else "None"

        if highest_severity != "None":
            if cd_score >= CODE_DENSITY_THRESHOLD and sev_conf < CONFIDENCE_THRESHOLD and ent_count < ENTITY_SAFEGUARD_COUNT:
                adjusted_severity = "None"
                final_threat_level = "Benign (Code-heavy, low confidence)"
                downgrade_reason = f"code_density={cd_score:.4f}, sev_conf={sev_conf:.3f}, entities={ent_count}"
                logging.info("Downgrading %s to None due to code-heavy heuristic (%s)", folder, downgrade_reason)
                is_suspicious = False
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

                novelty = compute_novelty(combined_text, seen_hashes_local)
                density = keyword_density(combined_text)
                noise_factor = 1 + max(0, len(combined_text.split()) / 1000 - density)
                relevance = max(0.4, 0.3 * density + 0.7 * sev_conf)
                score_val = compute_score(adjusted_severity, relevance, novelty, noise_factor)
                adjusted_severity = adjust_severity(adjusted_severity, score_val)
                final_threat_level = get_final_threat_level(adjusted_severity, score_val)
        else:
            if threat_conf > 0:
                novelty = compute_novelty(combined_text, seen_hashes_local)
                density = keyword_density(combined_text)
                noise_factor = 1 + max(0, len(combined_text.split()) / 1000 - density)
                relevance = max(0.2, 0.2 * density + 0.8 * threat_conf)
                score_val = compute_score("Medium", relevance, novelty, noise_factor)
                if score_val and score_val >= 1.5:
                    adjusted_severity = "Medium"
                    final_threat_level = get_final_threat_level(adjusted_severity, score_val)
                    is_suspicious = True

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

        # If after heuristics we determine nothing suspicious, skip insertion
        if adjusted_severity == "None" and not is_suspicious:
            scanned.add(folder_path)
            logging.info("%s | Final: None (skipped insertion) | Reason: %s", folder, downgrade_reason or "heuristic/whitelist")
            continue

        # Build simplified JSON doc with only the fields the user requested
        # Extract entities
        entities = extract_entities_simple(combined_text)

        # compute raw_score_for_norm for risk normalization
        if score_val is not None:
            raw_score_for_norm = score_val
        else:
            raw_score_for_norm = (sev_conf * 2.0) + threat_conf

        risk_score = safe_normalize_risk(raw_score_for_norm)

        # Prepare credentials: emails formatted as mailto markdown, usernames plain
        credentials_out = []
        for cred in entities.get("credentials", []):
            v = cred.get("value")
            if cred.get("type") == "email":
                credentials_out.append(f"[{v}](mailto:{v})")
            else:
                credentials_out.append(v)

        # Decide label: Threat if suspicious per models/heuristics, else NoThreat
        label_out = "Threat" if is_suspicious else "NoThreat"

        # model confidence: choose the best model confidence available (severity or threat)
        model_confidence = max(float(sev_conf or 0.0), float(threat_conf or 0.0))
        model_confidence = round(model_confidence, 3)

        # urgency and action
        urgency = urgency_from_risk(risk_score)
        action_suggestion = action_from_urgency(urgency)

        # timestamp
        timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

        # onsite_url - use folder identifier (same as earlier)
        onsite_url = folder

        simplified_doc = {
            "onsite_url": onsite_url,
            "label": label_out,
            "risk_score": int(risk_score),
            "urgency": urgency,
            "credentials": credentials_out,
            "action_suggestion": action_suggestion,
            "timestamp": timestamp,
            "model_confidence": float(model_confidence)
        }

        # Insert into DynamoDB
        if ddb_table is not None:
            try:
                # DynamoDB requires Decimal for floats — convert numeric fields explicitly
                item_to_put = simplified_doc.copy()
                # Convert floats to Decimal where present
                item_to_put["model_confidence"] = Decimal(str(item_to_put["model_confidence"]))
                # Put item (this will overwrite existing item with same partition key)
                ddb_table.put_item(Item=item_to_put)
                logging.info("Inserted item into DynamoDB table %s for %s", DDB_TABLE_NAME, onsite_url)
            except botocore.exceptions.ClientError as e:
                logging.warning("DynamoDB client error when inserting %s: %s", onsite_url, e)
            except Exception as e:
                logging.warning("Failed to insert into DynamoDB for %s: %s", onsite_url, e)
        else:
            logging.warning("DynamoDB table not initialized; skipping insertion for %s", onsite_url)

        results.append(simplified_doc)
        scanned.add(folder_path)

        logging.info(
            "%s | Risk:%d | Label:%s | ModelConf:%0.3f | Entities:%d",
            folder, int(risk_score), label_out, model_confidence, len(entities.get("credentials", []))
        )

    return results

def _simple_language_score(text):
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return round(min(1.0, alpha / max(1, len(text))), 3)

# ---------- Stage helpers ----------
def train_stage1():
    df = pd.read_csv(STAGE1_CSV)
    train_model(df, num_labels=2, save_dir=STAGE1_MODEL_DIR)

def train_stage2():
    df = pd.read_csv(STAGE2_CSV)
    train_model(df, num_labels=3, save_dir=STAGE2_MODEL_DIR)

def load_models_and_tokenizers():
    """
    Load tokenizers and models for stage1 (threat) and stage2 (severity).
    Returns tuple:
      (tokenizer_threat, model_threat, tokenizer_severity, model_severity)
    If a model/tokenizer cannot be loaded, returns None in its place.
    """
    tokenizer_threat = None
    model_threat = None
    tokenizer_severity = None
    model_severity = None

    # Stage1
    if os.path.exists(STAGE1_MODEL_DIR):
        try:
            tokenizer_threat = AutoTokenizer.from_pretrained(STAGE1_MODEL_DIR)
            model_threat = AutoModelForSequenceClassification.from_pretrained(STAGE1_MODEL_DIR).to(device)
            model_threat.eval()
            logging.info("Loaded Stage1 (threat) model from %s", STAGE1_MODEL_DIR)
        except Exception as e:
            logging.warning("Failed to load Stage1 model/tokenizer: %s", e)
            tokenizer_threat, model_threat = None, None
    else:
        logging.info("Stage1 model dir %s not found.", STAGE1_MODEL_DIR)

    # Stage2
    if os.path.exists(STAGE2_MODEL_DIR):
        try:
            tokenizer_severity = AutoTokenizer.from_pretrained(STAGE2_MODEL_DIR)
            model_severity = AutoModelForSequenceClassification.from_pretrained(STAGE2_MODEL_DIR).to(device)
            model_severity.eval()
            logging.info("Loaded Stage2 (severity) model from %s", STAGE2_MODEL_DIR)
        except Exception as e:
            logging.warning("Failed to load Stage2 model/tokenizer: %s", e)
            tokenizer_severity, model_severity = None, None
    else:
        logging.info("Stage2 model dir %s not found.", STAGE2_MODEL_DIR)

    return tokenizer_threat, model_threat, tokenizer_severity, model_severity

# ---------- Main (continuous scanning loop) ----------
def main():
    whitelist = load_whitelist()

    # Ensure training data files exist; if models missing, train them
    if not os.path.exists(STAGE1_CSV):
        raise FileNotFoundError(f"Stage1 CSV not found: {STAGE1_CSV}. Create or place it in project root.")
    if not os.path.exists(STAGE2_CSV):
        raise FileNotFoundError(f"Stage2 CSV not found: {STAGE2_CSV}. Create or place it in project root.")

    # Train if model dirs missing
    if not os.path.exists(STAGE1_MODEL_DIR):
        logging.info("Stage1 model not found. Training Stage1...")
        train_stage1()
    else:
        logging.info("Stage1 model found. Skipping training.")

    if not os.path.exists(STAGE2_MODEL_DIR):
        logging.info("Stage2 model not found. Training Stage2...")
        train_stage2()
    else:
        logging.info("Stage2 model found. Skipping training.")

    # Load models and tokenizers (once)
    tokenizer_threat, model_threat, tokenizer_severity, model_severity = load_models_and_tokenizers()

    if tokenizer_threat is None and tokenizer_severity is None:
        logging.warning("Neither stage1 nor stage2 tokenizers could be loaded. Exiting.")
        return []

    # Data types for detection (from stage2 CSV labels if exists)
    df_types = pd.read_csv(STAGE2_CSV)["label"].unique().tolist() if os.path.exists(STAGE2_CSV) else []

    # persistent set of scanned folder paths; this prevents re-processing the same folder path repeatedly
    scanned = set()

    logging.info("Starting continuous scanning loop. Base dir: 'scan'. Sleep between cycles: %s seconds", SCAN_SLEEP_SECONDS)
    try:
        while True:
            # Run single scan that uses both models (if available)
            try:
                new_results = scan_folders(
                    base_dir="scan",
                    threat_model=model_threat,
                    severity_model=model_severity,
                    tokenizer_threat=tokenizer_threat,
                    tokenizer_severity=tokenizer_severity,
                    scanned=scanned,
                    df_types=df_types,
                    whitelist=whitelist,
                    source_map=None,
                    search_keyword=None
                )
            except Exception as e:
                logging.exception("scan_folders raised an unexpected exception: %s", e)
                new_results = []

            if new_results:
                # write a separate results file for this scan cycle
                ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                out_file = os.path.join("logs", f"darkweb_results_{ts}.json")
                os.makedirs("logs", exist_ok=True)
                try:
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(new_results, f, indent=2)
                    logging.info("Saved %d new results to %s", len(new_results), out_file)
                except Exception as e:
                    logging.warning("Failed to write results file: %s", e)
            else:
                logging.debug("No new items found in this cycle.")

            # Sleep before next cycle
            time.sleep(SCAN_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting continuous scanning loop.")
    except Exception as e:
        logging.exception("Unexpected error in main loop: %s", e)

    # return scanned items summary (optional)
    logging.info("Total scanned folder count at exit: %d", len(scanned))
    return list(scanned)


if __name__ == "__main__":
    results = main()
    print("Exited. Scanned folders count:", len(results))
