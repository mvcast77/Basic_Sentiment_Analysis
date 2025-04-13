from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd


def initialize(xtrain, ytrain):

    data_frame = pd.read_csv(xtrain)
    y_result_df = pd.read_csv(ytrain)
    y_result = y_result_df.to_numpy()

    corpus = data_frame[['text']]
    corpus_list = [x for str in corpus['text'] if (x := str)]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus_list)
    matrix = matrix.toarray()

    N = matrix.shape[0]
    p = matrix.shape[1]

    matrix = np.c_[matrix, np.ones(N)]

    weights = np.ones((p+1, 1))

    return matrix, y_result, weights
