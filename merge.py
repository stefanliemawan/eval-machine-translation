import pandas as pd

dutch_df = pd.read_csv("result/v1/mbart50/translated_from_dutch.csv")
finnish_df = pd.read_csv("result/v1/mbart50/translated_from_finnish.csv")
df = finnish_df.merge(dutch_df)
df.drop(columns=["Unnamed: 0"], inplace=True)

print(df)

df.to_csv("result/v1/mbart50/translated_1.csv")
