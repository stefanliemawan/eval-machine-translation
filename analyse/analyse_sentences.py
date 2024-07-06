from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud

sentences_df = pd.read_csv("../dataset/sentences_v1.csv", index_col=0)

print(sentences_df)

english = sentences_df["english"].tolist()