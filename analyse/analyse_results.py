import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    "mbart-50",
    "nllb-200",
    "m2m-100",
]

BAR_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


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
        ax.legend(loc="lower right", fontsize="small", framealpha=0.5)


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

plt.clf()

# Compute average scores for each model
average_bleu_scores = bleu_scores.mean()
average_sacre_bleu_scores = sacre_bleu_scores.mean()
average_meteor_scores = meteor_bleu_scores.mean()

# Create a DataFrame for average scores
average_scores = pd.DataFrame(
    {
        "BLEU": average_bleu_scores,
        "SacreBLEU": average_sacre_bleu_scores,
        "METEOR": average_meteor_scores,
    }
)

fig, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(8, 10))

bar_width = 0.8
index = np.arange(len(average_scores.index))

# Plot average BLEU scores
ax4.bar(index, average_scores["BLEU"], bar_width, color=BAR_COLOURS)
ax4.set_ylabel("Average BLEU Score")
ax4.set_title("Average BLEU Scores by Model")
ax4.set_xticks(index)
ax4.set_xticklabels(average_scores.index, rotation=45, ha="right")

# Plot average SacreBLEU scores
ax5.bar(index, average_scores["SacreBLEU"], bar_width, color=BAR_COLOURS)
ax5.set_ylabel("Average SacreBLEU Score")
ax5.set_title("Average SacreBLEU Scores by Model")
ax5.set_xticks(index)
ax5.set_xticklabels(average_scores.index, rotation=45, ha="right")

# Plot average METEOR scores
ax6.bar(index, average_scores["METEOR"], bar_width, color=BAR_COLOURS)
ax6.set_ylabel("Average METEOR Score")
ax6.set_title("Average METEOR Scores by Model")
ax6.set_xticks(index)
ax6.set_xticklabels(average_scores.index, rotation=45, ha="right")

plt.tight_layout()
plt.savefig(f"./figures/average_metrics_bar.png")

plt.clf()

data = []

for lang in LANGS:
    for model in MODELS:
        sacre_bleu = metrics[model][lang]["sacre_bleu"]
        meteor = metrics[model][lang]["meteor"]
        data.append([lang, model, "SacreBLEU", sacre_bleu])
        data.append([lang, model, "METEOR", meteor])

df = pd.DataFrame(data, columns=["Language", "Model", "Metric", "Score"])


for lang in LANGS:
    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    subset_sacre_bleu = df[(df["Language"] == lang) & (df["Metric"] == "SacreBLEU")]
    sns.barplot(
        ax=axes[0], x="Model", y="Score", data=subset_sacre_bleu, palette=BAR_COLOURS
    )
    axes[0].set_title(f"{lang.capitalize()} - SacreBLEU")
    axes[0].set_ylim(0, 100)

    subset_meteor = df[(df["Language"] == lang) & (df["Metric"] == "METEOR")]
    sns.barplot(
        ax=axes[1], x="Model", y="Score", data=subset_meteor, palette=BAR_COLOURS
    )
    axes[1].set_title(f"{lang.capitalize()} - METEOR")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"./figures/{lang}_all_metrics.png")
    plt.close()
