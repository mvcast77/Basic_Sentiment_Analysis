from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

doc = "x_train.csv"

data_frame = pd.read_csv(doc)

corpus = data_frame[['text']]
corpus_list = [x for str in corpus['text'] if (x := str)]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus_list)
print(vectorizer.get_feature_names_out())
print(matrix.shape)

print(matrix)
