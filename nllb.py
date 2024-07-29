import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "facebook/nllb-200-3.3B"

MBART50_LANG_CODES = {
    "dutch": "nld_Latn",
    "finnish": "fin_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "hebrew": "he_IL",
    "italian": "ita_Latn",
    "japanese": "jpn_Jpan",
    "chinese": "zho_Hans",
    "polish": "pol_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "spanish": "spa_Latn",
    "turkish": "tur_Latn",
    "ukrainian": "ukr_Cyrl",
}

tqdm.pandas()

tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

df = pd.read_csv("dataset/sentences_v1.csv", index_col=0)


def translate_batch(batch):
    inputs = tokeniser(batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs
        )
    translations = tokeniser.batch_decode(translated_tokens, skip_special_tokens=True)
    print(batch)
    print(translations)
    asd

    return translations


batch_size = 1
# batch_size = 8
translated_df = df[["index", "english"]].copy()

for src_lang, src_lang_code in MBART50_LANG_CODES.items():
    print(f"Translating {src_lang}...")
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

    print(translated_df)
    translated_df.to_csv(f"result/{MODEL_NAME.split("/")[1]}.csv")

# [
#     "Ik moet gaan slapen.",
#     "Muiriel is nu 20 jaar oud.",
#     'Het wachtwoord is "Muiriel".',
#     "Ik ben zo terug.",
#     "Ik heb er geen woorden voor. | Woorden schieten me tekort.",
#     "Hier komt nooit een eind aan. | Dit zal nooit eindigen.",
#     "Ik weet gewoon niet wat ik moet zeggen... | Ik weet eenvoudig niet wat te zeggen...",
#     "Ik was in de bergen.",
# ]
# [
#     "Lo egin beharko nuke.",
#     "Muiriel nun estas dudekjara.",
#     'La pasvorto estas "Muiriel".',
#     "Berehala etorriko naiz.",
#     "I don't have the words. I'm at a loss for words.",
#     "Ez da inoiz amaituko.",
#     "I just don't know what to say... I just don't know what to say...",
#     "Mi estis en la montaro.",
# ]

# wrong config? translation seems weird
