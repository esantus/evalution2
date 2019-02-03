"""
This module includes a function to load word embeddings and run
several classifiers to predict the relation between words.

author: enrico santus
"""

import numpy as np
import pdb
import embeddings as emb

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def get_vec(w, embs, w2i):
    '''
    Return the embedding vector for word w or False
    '''
    return embs[w2i[w], :] if w in w2i else False


def combine(w1, w2, combination):
    '''
    Return the combination of two vectors:
       - concat
       - sum
       - mult
    '''
    if combination == 'concat':
        return np.concatenate((w1, w2))
    if combination == 'sum':
        return w1 + w2
    if combination == 'mult':
        return w1 * w2


def load_dataset(dataset, combinations, embs, w2i):
    '''
    Load the combined embeddings in the dataset and return
    them together with the dataset updated (i.e. without oov)
    '''
    X, Y = {}, {}
    ds = []
    for w1, w2, rel in dataset:
        w1_emb = get_vec(w1, embs, w2i)
        w2_emb = get_vec(w2, embs, w2i)
        if x != False and y != False:
            ds.append((w1, w2, rel))
            for combination in combinations:
                if combination not in X:
                    X[combination] = []
                    Y[combination] = []
                X[combination].append(combine(w1_emb, w2_emb, combination))
                Y[combination].append(rel)
    return X, Y, ds


def load_classifier(clf_name):
    '''
    Return one of the standard classifiers:
        - random_forest
        - k_neighbors
        - svc
        - decision_tree
        - mlp
        - ada_boost
    '''
    if clf_name == 'random_forest':
        return RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    if clf_name == 'k_neighbors':
        return KNeighborsClassifier(3)
    if clf_name == 'svc':
        return SVC(kernel="linear", C=0.025)
    if clf_name == 'decision_tree':
        return DecisionTreeClassifier(max_depth=5)
    if clf_name == 'mlp':
        return MLPClassifier(alpha=1)
    if clf_name == 'ada_boost':
        return AdaBoostClassifier()


def classify(train, dev, test, clfs=['random_forest', 'mlp', 'svc'], combinations=['concat', 'sum', 'mult'], emb_path='../data/embeddings/glove.6B.300d.txt', emb_dims=300):
    '''
    Load the embeddings and turn the datasets in a format that is
    compatible to the classifiers.
    '''
    embs, w2i = emb.load_embeddings(emb_path, emb_dims)

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = [], [], [], [], [], []

    X_train, Y_train, train = load_dataset(train, combinations, embs, w2i)
    X_dev, Y_dev, dev = load_dataset(dev, combinations, embs, w2i)
    X_test, Y_test, test = load_dataset(test, combinations, embs, w2i)

    for clf_name in clfs:
        clf = load_classifier(clf_name)

        for comb in X_train:
            clf.fit(X_train[comb], Y_train[comb])
            print('Classifier: {}\nScore is: {}'.format(clf_name, clf.score(X_test, y_test)))
            results[clf_name + ' ' + comb] = (test, clf.predict(X_test, y_test))

    return results


