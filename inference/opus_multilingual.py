import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"


LANGS = {
    "chinese": "zh",
    "dutch": "nl",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "hebrew": "he",
    "italian": "it",
    "japanese": "ja",
    "polish": "pl",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
    "ukrainian": "uk",
}

tqdm.pandas()

df = pd.read_csv("../dataset/sentences_v2.csv", index_col=0)


def translate_batch(tokeniser, batch_input):
    inputs = tokeniser(
        batch_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        translated_tokens = model.generate(**inputs)
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


batch_size = 4
translated_df = df[["index", "english"]].copy()
timing_df = pd.DataFrame()

for src_lang, src_lang_code in LANGS.items():
    print(f"Translating {src_lang}...")
    start = time.time()

    model = MarianMTModel.from_pretrained(MODEL_NAME)
    tokeniser = MarianTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=src_lang_code,
        tgt_lang="en",
    )

    translated_texts = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_input = df[src_lang][i : i + batch_size].tolist()
        translations = translate_batch(tokeniser, batch_input)
        translated_texts.extend(translations)

    translated_df[f"translated_from_{src_lang}"] = translated_texts

    end = time.time()
    timing_df[f"translated_from_{src_lang}"] = [end - start]

    translated_df.to_csv(f"result/opus.csv")
    timing_df.to_csv(f"result/opus_timings.csv")


# not entirely sure how to use the mt model
