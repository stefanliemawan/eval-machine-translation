import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MBartTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

MBART50_LANG_CODES = {
    # "dutch": "nl_XX",
    # "finnish": "fi_FI",
    "french": "fr_XX",
    "german": "de_DE",
    "hebrew": "he_IL",
    "italian": "it_IT",
    "japanese": "ja_XX",
    "mandarin": "zh_CN",
    "polish": "pl_PL",
    "portuguese": "pt_XX",
    "russian": "ru_RU",
    "spanish": "es_XX",
    "turkish": "tr_TR",
    "ukrainian": "uk_UA",
}

tqdm.pandas()

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
tokeniser = MBart50Tokenizer.from_pretrained(MODEL_NAME)

df = pd.read_csv("dataset/sentences_v1.csv", index_col=0)

print(df)


def translate_batch(batch):
    inputs = tokeniser(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokeniser.lang_code_to_id["en_XX"]
        )
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


batch_size = 8
translated_df = df[["index", "english"]].copy()

for src_lang in MBART50_LANG_CODES.keys():
    print(f"Translating {src_lang}...")
    translated_texts = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[src_lang][i : i + batch_size].tolist()
        translations = translate_batch(batch)
        translated_texts.extend(translations)

    translated_df[f"translated_from_{src_lang}"] = translated_texts

    print(translated_df)
    translated_df.to_csv("result/v1/mbart50/translated.csv")

# seems like some language still translated to non_english?