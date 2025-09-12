"""
GPU-based contextual augmentation for darkweb_data_stage1.csv
- Uses high-quality transformer embeddings (e.g. RoBERTa) for substitution
- Runs on GPU if available
- Preserves label and optional "type" column
- Logs summary stats
"""

import logging
import random
import re
import pandas as pd
import torch
import nlpaug.augmenter.word as naw

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# CONFIG
# -------------------------
AUG_PER_ROW = 3
INPUT_FILE = "darkweb_data_stage2.csv"
OUTPUT_FILE = "darkweb_data_stage2_augmented.csv"
RANDOM_SEED = 42
BATCH_SIZE = 16  # keep small to fit GPU VRAM
MODEL_NAME = "roberta-base"  # can be 'roberta-large' if you want max quality

random.seed(RANDOM_SEED)

# -------------------------
# Check device
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# -------------------------
# Load dataframe
# -------------------------
df = pd.read_csv(INPUT_FILE)
if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Input CSV must contain at least 'text' and 'label' columns.")

total_input_rows = len(df)

# -------------------------
# Create contextual augmenter
# -------------------------
logging.info(f"Loading contextual augmenter: {MODEL_NAME}")
aug = naw.ContextualWordEmbsAug(
    model_path=MODEL_NAME,
    action="substitute",
    device=DEVICE,
    top_k=50
)

# -------------------------
# Main augmentation loop
# -------------------------
augmented_rows = []
skipped_count = 0

logging.info(f"Starting augmentation: rows={total_input_rows}, aug_per_row={AUG_PER_ROW}")

for i in range(0, total_input_rows, BATCH_SIZE):
    batch = df.iloc[i:i + BATCH_SIZE]
    for _, row in batch.iterrows():
        text = str(row["text"]).strip()
        label = row["label"]
        row_type = row["type"] if "type" in df.columns else None

        if not text or len(text.split()) < 2:
            continue

        created = 0
        tries = 0
        while created < AUG_PER_ROW and tries < AUG_PER_ROW * 2:
            tries += 1
            try:
                augmented = aug.augment(text, n=1)
                if isinstance(augmented, list):
                    augmented = augmented[0]
                augmented = re.sub(r"\s+", " ", augmented).strip()

                if augmented.lower() != text.lower():
                    new_row = {"text": augmented, "label": label}
                    if row_type is not None:
                        new_row["type"] = row_type
                    augmented_rows.append(new_row)
                    created += 1
            except Exception as e:
                logging.debug(f"Augmentation failed for row: {e}")
                skipped_count += 1
                break

# -------------------------
# Combine and deduplicate
# -------------------------
df_aug = pd.DataFrame(augmented_rows)
if not df_aug.empty:
    df_combined = pd.concat([df, df_aug], ignore_index=True)
    df_combined.drop_duplicates(subset=["text", "label"], keep="first", inplace=True)
else:
    df_combined = df.copy()

# -------------------------
# Save output
# -------------------------
df_combined.to_csv(OUTPUT_FILE, index=False)
logging.info("Augmentation finished.")
logging.info(f"Original rows: {total_input_rows}")
logging.info(f"Augmented rows created: {len(df_combined) - total_input_rows}")
logging.info(f"Total rows after augmentation: {len(df_combined)}")
logging.info(f"Skipped augmentation attempts (approx): {skipped_count}")
logging.info(f"Saved augmented file to: {OUTPUT_FILE}")
