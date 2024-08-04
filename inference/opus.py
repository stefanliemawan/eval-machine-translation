import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

MODEL_DICT = {
    "chinese": "Helsinki-NLP/opus-mt-zh-en",
    "dutch": "Helsinki-NLP/opus-mt-nl-en",
    "finnish": "Helsinki-NLP/opus-mt-fi-en",
    "french": "Helsinki-NLP/opus-mt-fr-en",
    "german": "Helsinki-NLP/opus-mt-de-en",
    "hebrew": "tiedeman/opus-mt-he-en",
    "italian": "Helsinki-NLP/opus-mt-it-en",
    "japanese": "Helsinki-NLP/opus-mt-ja-en",
    "polish": "Helsinki-NLP/opus-mt-pl-en",
    "russian": "Helsinki-NLP/opus-mt-ru-en",
    "spanish": "Helsinki-NLP/opus-mt-es-en",
    "turkish": "Helsinki-NLP/opus-mt-tr-en",
    "ukrainian": "Helsinki-NLP/opus-mt-uk-en",
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

for src_lang, model_name in MODEL_DICT.items():
    print(f"Translating {src_lang}...")
    start = time.time()

    model = MarianMTModel.from_pretrained(model_name)
    tokeniser = MarianTokenizer.from_pretrained(model_name)

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
