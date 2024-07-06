import pandas as pd

langs_df = pd.read_csv("result/v1/mbart50/translated.csv")
dutch_fin_df = pd.read_csv("result/v1/mbart50/translated_dutch_finnish.csv")
print(langs_df)
df = dutch_fin_df.merge(langs_df, how="outer", on=["index", "english"])

df.drop(columns=["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)

print(df)

df.to_csv("result/v1/mbart50/translated_1.csv")
