import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MBartTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

LANG_CODES = {
    "chinese": "zh_CN",
    "dutch": "nl_XX",
    "finnish": "fi_FI",
    "french": "fr_XX",
    "german": "de_DE",
    "hebrew": "he_IL",
    "italian": "it_IT",
    "japanese": "ja_XX",
    "polish": "pl_PL",
    "russian": "ru_RU",
    "spanish": "es_XX",
    "turkish": "tr_TR",
    "ukrainian": "uk_UA",
}


model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

df = pd.read_csv("../dataset/sentences_v2.csv", index_col=0)
tqdm.pandas()


def translate_batch(tokeniser, batch_input):
    inputs = tokeniser(
        batch_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokeniser.lang_code_to_id["en_XX"]
        )
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


batch_size = 4
translated_df = df[["index", "english"]].copy()
timing_df = pd.DataFrame()

for src_lang, src_lang_code in LANG_CODES.items():
    print(f"Translating {src_lang}...")
    start = time.time()

    tokeniser = MBart50Tokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=src_lang_code,
        tgt_lang="en_XX",
    )
    translated_texts = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_input = df[src_lang][i : i + batch_size].tolist()
        translations = translate_batch(tokeniser, batch_input)
        translated_texts.extend(translations)

    translated_df[f"translated_from_{src_lang}"] = translated_texts

    end = time.time()
    timing_df[f"translated_from_{src_lang}"] = [end - start]

    translated_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}.csv")
    timing_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}_timings.csv")

