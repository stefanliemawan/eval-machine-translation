import json

import nltk
import numpy as np
import pandas as pd
import sacrebleu
from nltk.translate import meteor_score
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm

# nltk.download("punkt_tab")
# nltk.download("wordnet")

SRC_LANGS = [
    "chinese",
    "dutch",
    "finnish",
    "french",
    "german",
    "hebrew",
    "italian",
    "japanese",
    "polish",
    "russian",
    "spanish",
    "turkish",
    "ukrainian",
]

PATHS = [
    "opus",
    "mbart-large-50-many-to-many-mmt",
    "nllb-200-distilled-600M",
    "m2m100_418M",
]


def calculate_metric(reference, candidate):
    ref_tokenised = [nltk.word_tokenize(ref) for ref in reference]
    cand_tokenised = [nltk.word_tokenize(cand) for cand in candidate]

    meteor = np.mean(
        [
            meteor_score.single_meteor_score(ref, cand)
            for ref, cand in zip(ref_tokenised, cand_tokenised)
        ]
    )

    sacre_bleu = sacrebleu.corpus_bleu(candidate, [reference]).score

    bleu = corpus_bleu(
        ref_tokenised, cand_tokenised, smoothing_function=SmoothingFunction().method7
    )

    return bleu, sacre_bleu, meteor


metrics = {}

for path in PATHS:
    print(f"Calculating {path}...")
    df = pd.read_csv(f"./result/{path}.csv", index_col=0)
    metrics[path] = {}

    reference = df["english"].tolist()

    for src_lang in tqdm(SRC_LANGS):
        candidate = df[f"translated_from_{src_lang}"].tolist()
        bleu, sacre_blue, meteor = calculate_metric(reference, candidate)
        metrics[path][src_lang] = {
            "bleu": bleu,
            "sacre_bleu": sacre_blue,
            "meteor": meteor,
        }


print(metrics)


with open("./result/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4, sort_keys=True)
