import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd


def initialize(xtrain, ytrain, xtest):

    data_frame = pd.read_csv(xtrain)
    data_frame_test = pd.read_csv(xtest)
    y_result_df = pd.read_csv(ytrain)
    y_result = y_result_df.to_numpy()

    corpus = data_frame[['text']]
    corpus_test = data_frame_test[['text']]
    corpus_list = [x for str in corpus['text'] if (x := str)]
    N = len(corpus_list)
    corpus_list_test = [x for str in corpus_test['text'] if (x := str)]
    N_test = len(corpus_list_test)
    corpus_list = corpus_list + corpus_list_test
    print(N)
    print(N_test)
    print(N + N_test)
    print(len(corpus_list))

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus_list)
    matrix = matrix.toarray()

    matrix, matrix_test = np.split(matrix, [N]);
    p = matrix.shape[1]

    matrix = np.c_[matrix, np.ones(N)]
    matrix_test = np.c_[matrix_test, np.ones(N_test)]

    weights = np.full((p+1, 1), 0.5)

    print(matrix.shape)
    print(matrix_test.shape)
    print(y_result.shape)
    print(weights.shape)

    return matrix, y_result, weights, matrix_test

def outputter(ypredictions, y_file):
    fobj = open(y_file, 'w')
    for predictions in ypredictions:
        fobj.write(str(predictions[0]) + "\n")
    fobj.close()
