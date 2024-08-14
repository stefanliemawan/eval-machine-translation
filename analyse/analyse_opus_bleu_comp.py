import matplotlib.pyplot as plt

# Data from the table
languages = [
    "Chinese",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Hebrew",
    "Italian",
    "Japanese",
    "Polish",
    "Russian",
    "Spanish",
    "Turkish",
    "Ukrainian",
]
sacrebleu_scores = [
    59.5,
    69.8,
    66.6,
    69.8,
    69.7,
    66.5,
    74.1,
    63.1,
    61.9,
    66.7,
    71.4,
    72.6,
    75.4,
]
reported_bleu_scores = [
    36.1,
    60.9,
    53.4,
    57.5,
    55.4,
    52.0,
    70.9,
    41.7,
    54.9,
    61.1,
    59.6,
    63.5,
    64.1,
]
# Reversing the data so that Chinese is on top and Ukrainian is on the bottom
languages_reversed = languages[::-1]
sacrebleu_scores_reversed = sacrebleu_scores[::-1]
reported_bleu_scores_reversed = reported_bleu_scores[::-1]

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 7))

# Plot SacreBLEU scores
bars1 = ax.barh(
    languages_reversed,
    sacrebleu_scores_reversed,
    color="skyblue",
    edgecolor="black",
    label="Result SacreBLEU",
)

# Plot Reported BLEU scores
for i, score in enumerate(reported_bleu_scores_reversed):
    if score is not None:
        ax.barh(
            languages_reversed[i],
            score,
            color="orange",
            edgecolor="black",
            alpha=0.6,
            label="Reported BLEU" if i == 0 else "",
        )

# Adding labels and title
ax.set_xlabel("BLEU Score")
ax.set_title("Comparison of SacreBLEU and Reported BLEU Scores by Language")

# Creating legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left")

# Display the plot
plt.tight_layout()

plt.savefig(f"./figures/opus_bleu_comparison.png")
