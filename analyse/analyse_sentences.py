import os
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from textblob import TextBlob
from wordcloud import WordCloud

os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_STATE = 42

sentences_df = pd.read_csv("../dataset/sentences_v2.csv", index_col=0)
english = sentences_df["english"].tolist()

LANG = [
    "english",
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


def calculate_len(sentence):
    if "|" in sentence:
        sentence = sentence.split("|")
        return min([len(nltk.word_tokenize(x)) for x in sentence])
    else:
        return len(nltk.word_tokenize(sentence))


def analyse_sentence_length(df):
    plt.clf()
    plt.figure(figsize=(10, 6))

    for lang in LANG:
        if lang in ["chinese", "japanese"]:
            df[f"{lang}_length"] = df[lang].apply(len)
        else:
            df[f"{lang}_length"] = df[lang].apply(calculate_len)

        plt.hist(df[f"{lang}_length"], bins=20, alpha=0.3, label=lang)

    plt.legend(loc="upper right")
    plt.title("Word Count Distribution")
    plt.ylabel("Word Count")
    plt.ylabel("Frequency")

    plt.savefig("./figures/word_count_hist.png")


def analyse_sentence_length_2(df):
    plt.clf()
    plt.figure(figsize=(15, 6))

    # Prepare data for box plot
    lengths = []
    languages = []
    for lang in LANG:
        if lang in ["chinese", "japanese"]:
            df[f"{lang}_length"] = df[lang].apply(len)
        else:
            df[f"{lang}_length"] = df[lang].apply(calculate_len)

        lengths.extend(df[f"{lang}_length"])
        languages.extend([lang] * len(df))

    # Create a DataFrame for the lengths and languages
    length_df = pd.DataFrame({"Word Count": lengths, "Language": languages})

    # Box plot for sentence lengths
    sns.boxplot(x="Language", y="Word Count", data=length_df)

    plt.title("Word Count Distribution")
    plt.ylabel("Word Count")
    plt.xlabel(None)

    plt.savefig("./figures/word_count_box.png")


def generate_word_cloud(df):
    wordcloud = WordCloud(width=5000, height=3000, background_color="white").generate(
        " ".join(df["english"])
    )

    # plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"figures/wordcloud.png")


def analyse_embedding(df):
    plt.clf()
    fig, axs = plt.subplots(3, 5, figsize=(60, 30))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings_eng = model.encode(df["english"].tolist())

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    embeddings_eng_2d = tsne.fit_transform(embeddings_eng)
    embeddings_eng_2d_df = pd.DataFrame(embeddings_eng_2d, columns=["x", "y"])

    i, j = 0, 0

    for lang in LANG[1:]:
        ax = axs[i, j]

        sns.scatterplot(x="y", y="x", data=embeddings_eng_2d_df, ax=ax)

        embeddings_lang = model.encode(df[lang].tolist())

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        embeddings_lang_2d = tsne.fit_transform(embeddings_lang)
        embeddings_lang_2d_df = pd.DataFrame(embeddings_lang_2d, columns=["x", "y"])

        sns.scatterplot(x="y", y="x", data=embeddings_lang_2d_df, ax=ax)
        ax.set_title(f"eng-{lang}")

        if j < 4:
            j += 1
        else:
            i += 1
            j = 0

    plt.tight_layout()

    plt.savefig(f"figures/embedding.png")


# analyse_sentence_length(sentences_df)
# analyse_sentence_length_2(sentences_df)
# generate_word_cloud(sentences_df)
analyse_embedding(sentences_df)
