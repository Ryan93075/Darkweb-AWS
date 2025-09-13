#!/usr/bin/env python3
"""
augment_with_nlpaug.py
Usage:
  python augment_with_nlpaug.py --input darkweb_data_stage1_augmented.csv --out augmented_stage1.csv --n 3

Creates n augmented variants per row using nlpaug.
"""
import argparse
import csv
import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.audio as naa  # not used, but shows options
import nlpaug.augmenter.word as naw2

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
    # contextual (BERT) augmenter: requires transformers model downloads; slower but higher quality
    # we use small number of augmenters and fall back to synonyms/char-level for speed
    try:
        contextual = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
    except Exception as e:
        print("Contextual augmenter unavailable (transformer download may fail). Falling back to synonyms.")
        contextual = None

    synonym_aug = naw.SynonymAug(aug_src='wordnet')  # requires nltk wordnet
    char_swap = nac.KeyboardAug()
    random_swap = naw.RandomWordAug(action="swap")
    # you can add other augmenters like BackTranslationAug (slow), or embedding-based augs
    return contextual, synonym_aug, char_swap, random_swap

def augment_text(text, contextual, synonym_aug, char_swap, random_swap):
    aug_texts = set()
    # 1) contextual (if available)
    if contextual:
        try:
            t1 = contextual.augment(text)
            if isinstance(t1, list):
                t1 = t1[0]
            aug_texts.add(t1)
        except Exception:
            pass
    # 2) synonyms
    try:
        t2 = synonym_aug.augment(text)
        if isinstance(t2, list):
            t2 = t2[0]
        aug_texts.add(t2)
    except Exception:
        pass
    # 3) keyboard char noise
    try:
        t3 = char_swap.augment(text)
        if isinstance(t3, list):
            t3 = t3[0]
        aug_texts.add(t3)
    except Exception:
        pass
    # 4) random swap
    try:
        t4 = random_swap.augment(text)
        if isinstance(t4, list):
            t4 = t4[0]
        aug_texts.add(t4)
    except Exception:
        pass

    # ensure we return some variants; at least original if nothing succeeded
    if not aug_texts:
        return [text]
    else:
        return list(aug_texts)

def main(input_file, output_file, n_aug=2):
    rows = load_csv(input_file)
    contextual, synonym_aug, char_swap, random_swap = build_augmenters()

    out_rows = []
    for text,label in rows:
        out_rows.append((text,label))
        # create n_aug augmentations
        tries = 0
        generated = 0
        while generated < n_aug and tries < n_aug * 4:
            tries += 1
            variants = augment_text(text, contextual, synonym_aug, char_swap, random_swap)
            if not variants:
                break
            # pick one randomly
            v = random.choice(variants)
            # simple sanity: avoid duplicates and too-short outputs
            if v and v != text and len(v.split()) >= max(3, len(text.split())//3):
                out_rows.append((v, label))
                generated += 1
    save_csv(output_file, out_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--n", type=int, default=2, help="Augmentations per row")
    args = parser.parse_args()
    main(args.input, args.out, args.n)
