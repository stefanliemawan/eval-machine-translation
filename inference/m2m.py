import time

import pandas as pd
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_NAME = "facebook/m2m100_418M"
tokeniser = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)

df = pd.read_csv("../dataset/sentences_v2.csv", index_col=0)
tqdm.pandas()

LANG_CODES = {
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


def translate_batch(tokeniser, batch_input):
    inputs = tokeniser(
        batch_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokeniser.get_lang_id("en")
    )
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)

    return translations


batch_size = 4
translated_df = df[["index", "english"]].copy()
timing_df = pd.DataFrame()

for src_lang, src_lang_code in LANG_CODES.items():
    print(f"Translating {src_lang}...")
    start = time.time()

    tokeniser = M2M100Tokenizer.from_pretrained(
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

    translated_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}.csv")
    timing_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}_timings.csv")


# use vsc
