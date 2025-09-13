#!/usr/bin/env python3
"""
augment_both_stages.py
Augments Stage1 and Stage2 datasets using NLP augmenters.
Usage:
    python augment_both_stages.py --stage1 darkweb_data_stage1_augmented.csv \
                                  --stage2 darkweb_data_stage2_augmented.csv \
                                  --n 3
"""

import argparse
import csv
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import os
import nltk

# Make sure wordnet is downloaded
nltk.download('wordnet')

random.seed(42)

def load_csv(input_file):
    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["text"], row["label"]))
    return rows

def save_csv(output_file, rows):
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text","label"])
        for t,l in rows:
            w.writerow([t, l])
    print(f"Wrote {len(rows)} rows to {output_file}")

def build_augmenters():
    synonym_aug = naw.SynonymAug(aug_src='wordnet')  # synonyms
    char_swap = nac.KeyboardAug()                     # keyboard typos
    random_swap = naw.RandomWordAug(action="swap")   # swap words
    return synonym_aug, char_swap, random_swap

def augment_text(text, synonym_aug, char_swap, random_swap):
    aug_texts = set()
    # synonym replacement
    try:
        t1 = synonym_aug.augment(text)
        if isinstance(t1, list):
            t1 = t1[0]
        aug_texts.add(t1)
    except Exception:
        pass
    # char swap
    try:
        t2 = char_swap.augment(text)
        if isinstance(t2, list):
            t2 = t2[0]
        aug_texts.add(t2)
    except Exception:
        pass
    # random swap
    try:
        t3 = random_swap.augment(text)
        if isinstance(t3, list):
            t3 = t3[0]
        aug_texts.add(t3)
    except Exception:
        pass
    # fallback
    if not aug_texts:
        return [text]
    else:
        return list(aug_texts)

def augment_dataset(input_file, n_aug=2):
    rows = load_csv(input_file)
    synonym_aug, char_swap, random_swap = build_augmenters()
    out_rows = []

    for text,label in rows:
        out_rows.append((text,label))  # keep original
        generated = 0
        tries = 0
        while generated < n_aug and tries < n_aug * 4:
            tries += 1
            variants = augment_text(text, synonym_aug, char_swap, random_swap)
            if not variants:
                break
            v = random.choice(variants)
            if v and v != text and len(v.split()) >= max(3, len(text.split())//3):
                out_rows.append((v, label))
                generated += 1

    output_file = os.path.splitext(input_file)[0] + "_augmented.csv"
    save_csv(output_file, out_rows)

def main(stage1_file, stage2_file, n_aug):
    print("Augmenting Stage1 dataset...")
    augment_dataset(stage1_file, n_aug)
    print("Augmenting Stage2 dataset...")
    augment_dataset(stage2_file, n_aug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", required=True, help="Stage1 CSV file")
    parser.add_argument("--stage2", required=True, help="Stage2 CSV file")
    parser.add_argument("--n", type=int, default=2, help="Augmentations per row")
    args = parser.parse_args()
    main(args.stage1, args.stage2, args.n)
