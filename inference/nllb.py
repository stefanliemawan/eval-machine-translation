import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# MODEL_NAME = "facebook/nllb-200-3.3B"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANG_CODES = {
    "chinese": "zho_Hans",
    "dutch": "nld_Latn",
    "finnish": "fin_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "hebrew": "he_IL",
    "italian": "ita_Latn",
    "japanese": "jpn_Jpan",
    "polish": "pol_Latn",
    "russian": "rus_Cyrl",
    "spanish": "spa_Latn",
    "turkish": "tur_Latn",
    "ukrainian": "ukr_Cyrl",
}

tqdm.pandas()

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

df = pd.read_csv("../dataset/sentences_v2.csv", index_col=0)


def translate_batch(batch):
    inputs = tokeniser(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs
        )
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


batch_size = 4
translated_df = df[["index", "english"]].copy()
timing_df = pd.DataFrame()

for src_lang, src_lang_code in LANG_CODES.items():
    print(f"Translating {src_lang}...")
    start = time.time()

    tokeniser = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=src_lang_code,
        tgt_lang="eng_Latn",
    )
    translated_texts = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[src_lang][i : i + batch_size].tolist()
        translations = translate_batch(batch)
        translated_texts.extend(translations)

    translated_df[f"translated_from_{src_lang}"] = translated_texts

    end = time.time()
    timing_df[f"translated_from_{src_lang}"] = [end - start]

    translated_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}.csv")
    timing_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}_timing.csv")
