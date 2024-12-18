import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

movies = pd.read_csv("C:\Users\EEmreÖZ\Documents\Coding\BgData\movies.csv")

def clean_title(title):
    re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(clean_title)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

title = 'Harry Potter'
title = clean_title(title)
query_vec = vectorizer.transform([title])
similarity = cosine_similarity(query_vec, tfidf).flatten()