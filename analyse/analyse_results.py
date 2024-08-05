import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from textblob import TextBlob
from wordcloud import WordCloud

LANGS = [
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

MODELS = [
    "opus",
    "mbart-large-50-many-to-many-mmt",
    "nllb-200-distilled-600M",
    "m2m100_418M",
]

with open("../inference/result/metrics.json", "r") as f:
    metrics = json.loads(f.read())


def plot_scores(dataframe, metric_name, ax, add_legend=False):
    bar_width = 0.2
    index = np.arange(len(dataframe.index))

    for i, model in enumerate(dataframe.columns):
        ax.bar(index + i * bar_width, dataframe[model], bar_width, label=model)

    ax.set_ylabel(f"{metric_name} Score")
    ax.set_title(f"{metric_name} Scores by Language and Model")
    ax.set_xticks(index + bar_width * (len(dataframe.columns) - 1) / 2)
    ax.set_xticklabels(dataframe.index, rotation=45, ha="right")
    if add_legend:
        ax.legend(loc="upper right", fontsize="small")


bleu_scores = pd.DataFrame(index=LANGS)
sacre_bleu_scores = pd.DataFrame(index=LANGS)
meteor_bleu_scores = pd.DataFrame(index=LANGS)

for model in MODELS:
    bleu_scores[model] = [metrics[model][lang]["bleu"] for lang in LANGS]
    sacre_bleu_scores[model] = [metrics[model][lang]["sacre_bleu"] for lang in LANGS]
    meteor_bleu_scores[model] = [metrics[model][lang]["meteor"] for lang in LANGS]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

plot_scores(bleu_scores, "BLEU", ax1, add_legend=True)

plot_scores(sacre_bleu_scores, "SacreBLEU", ax2)

plot_scores(meteor_bleu_scores, "METEOR", ax3)

plt.tight_layout()
plt.savefig(f"./figures/metrics_bar.png")
