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

MODELS = [
    "opus",
    "mbart-50",
    "nllb-200",
    "m2m-100",
]

opus_timings = pd.read_csv("../inference/result/opus_timings.csv", index_col=0)
mbart50_timings = pd.read_csv(
    "../inference/result/mbart-large-50-many-to-many-mmt_timings.csv", index_col=0
)
m2m100_timings = pd.read_csv("../inference/result/m2m100_418M_timings.csv", index_col=0)
nllb200_timings = pd.read_csv(
    "../inference/result/nllb-200-distilled-600M_timing.csv", index_col=0
)


# Combine the data into a single DataFrame
combined_timings = pd.DataFrame(
    {
        "opus": opus_timings.iloc[0],
        "mbart50": mbart50_timings.iloc[0],
        "m2m100": m2m100_timings.iloc[0],
        "nllb": nllb200_timings.iloc[0],
    }
)

# Plot the differences in inference timing for every language
plt.figure(figsize=(14, 8))

languages = combined_timings.index

for model in combined_timings.columns:
    plt.plot(languages, combined_timings[model], marker="o", label=model)

plt.xlabel("Language")
plt.ylabel("Inference Time (seconds)")
plt.title("Inference Timing Comparison for Different Models")
plt.xticks(rotation=45, ha="right")
plt.legend()
# plt.grid(True)
plt.tight_layout()

plt.savefig("./figures/inference_timings.png")
