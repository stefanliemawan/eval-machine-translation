import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MBartTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

MBART50_LANG_CODES = {
    "chinese": "zh_CN",
    "dutch": "nl_XX",
    "finnish": "fi_FI",
    "french": "fr_XX",
    "german": "de_DE",
    "hebrew": "he_IL",
    "italian": "it_IT",
    "japanese": "ja_XX",
    "polish": "pl_PL",
    # "portuguese": "pt_XX",
    "russian": "ru_RU",
    "spanish": "es_XX",
    "turkish": "tr_TR",
    "ukrainian": "uk_UA",
}

tqdm.pandas()

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

df = pd.read_csv("dataset/sentences_v1.csv", index_col=0)


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


batch_size = 8
translated_df = df[["index", "english"]].copy()

for src_lang, src_lang_code in MBART50_LANG_CODES.items():
    print(f"Translating {src_lang}...")
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
#     "I have to go to sleep.",
#     "Muiriel is now 20 years old.",
#     'The password is "Muiriel."',
#     "I'm so back.",
#     "I don't have words for it. | Words scare me.",
#     "This is never going to end. | This is never going to end.",
#     "I just don't know what to say. | I just don't know what to say.",
#     "I was in the mountains.",
# ]
