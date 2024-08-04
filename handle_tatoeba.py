import pandas as pd
from tqdm import tqdm


def read_df(language, tsv_path):
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        on_bad_lines="skip",
    )
    df.columns = ["index", "english", "index_target", language]
    df.drop(columns="index_target", inplace=True)

    return df


path = "original_dataset/tatoeba"

language_list = [
    (
        "chinese",
        f"{path}/Sentence pairs in English-Mandarin Chinese - 2024-07-05.tsv",
    ),
    (
        "dutch",
        f"{path}/Sentence pairs in English-Dutch - 2024-07-05.tsv",
    ),
    (
        "finnish",
        f"{path}/Sentence pairs in English-Finnish - 2024-07-05.tsv",
    ),
    (
        "french",
        f"{path}/Sentence pairs in English-French - 2024-07-05.tsv",
    ),
    (
        "german",
        f"{path}/Sentence pairs in English-German - 2024-07-05.tsv",
    ),
    (
        "hebrew",
        f"{path}/Sentence pairs in English-Hebrew - 2024-07-05.tsv",
    ),
    (
        "hungarian",
        f"{path}/Sentence pairs in English-Hungarian - 2024-07-05.tsv",
    ),
    (
        "italian",
        f"{path}/Sentence pairs in English-Italian - 2024-07-05.tsv",
    ),
    (
        "japanese",
        f"{path}/Sentence pairs in English-Japanese - 2024-07-05.tsv",
    ),
    (
        "polish",
        f"{path}/Sentence pairs in English-Polish - 2024-07-05.tsv",
    ),
    (
        "russian",
        f"{path}/Sentence pairs in English-Russian - 2024-07-05.tsv",
    ),
    (
        "spanish",
        f"{path}/Sentence pairs in English-Spanish - 2024-07-05.tsv",
    ),
    (
        "turkish",
        f"{path}/Sentence pairs in English-Turkish - 2024-07-05.tsv",
    ),
    (
        "ukrainian",
        f"{path}/Sentence pairs in English-Ukrainian - 2024-07-05.tsv",
    ),
]

main_df = pd.read_csv(
    "original_dataset/tatoeba/eng_sentences.tsv",
    sep="\t",
    header=None,
    on_bad_lines="skip",
)
main_df.columns = ["index", "language", "english"]
main_df.drop(columns=["language"], inplace=True)

for language, tsv_path in tqdm(language_list):
    lang_df = read_df(language, tsv_path)

    # lang_df_grouped = lang_df.groupby("index")[language].apply(
    #     lambda x: " | ".join(x.dropna().astype(str))
    # )
    lang_df_grouped = lang_df.groupby("index")[language].first()

    main_df = pd.merge(
        main_df,
        lang_df_grouped,
        on="index",
        how="outer",
    )


main_df.dropna(inplace=True)
print(main_df)

main_df.to_csv(f"dataset/sentences_v2.csv", index=True)
