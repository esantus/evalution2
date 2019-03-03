"""
This module includes a function to load word embeddings and run
several classifiers to predict the relation between words.

author: enrico santus
"""

import embeddings as emb
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_vec(w, embs, w2i):
    '''
    Return the embedding vector for word w or empty vector
    '''
    emb = []
    count = 0
    for c in w:
        if c in w2i:
            count += 1

        if emb == []:
            emb = embs[w2i[c]] if c in w2i else np.array([0] * len(embs[0]))
        else:
            emb += embs[w2i[c]] if c in w2i else np.array([0] * len(embs[0]))

    if count != 0:
        emb /= count
    else:
        print('No embeddings were found')

    print(emb, count)
    return emb


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
        if w1_emb != np.array([]) and w2_emb != np.array([]):
            ds.append([w1, w2, rel])
            for combination in combinations:
                if combination not in X:
                    X[combination] = []
                    Y[combination] = []
                X[combination].append(combine(w1_emb, w2_emb, combination))
                Y[combination].append(rel)
        else:
            print('Warning Out-Of-Vocabulary (line removed from the dataset): {}, {}, {}'.format(w1, w2, rel))
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


def print_predictions(y_true, y_pred, N=7):
    print("GROUND TRUTH\t\tPREDICTION")
    for ground_truth, pred in zip(y_true[:N], y_pred[:N]):
        print("{}\t\t{}".format(ground_truth, pred))


def classify(train, dev, test, clfs=['random_forest', 'mlp', 'svc'],
             combinations=['concat', 'sum', 'mult'],
             emb_path='..\data\embeddings\glove.6B.300d.txt', emb_dims=300):
    def classify(train, dev, test, clfs=['random_forest', 'mlp', 'svc'],
                 combinations=['concat', 'sum', 'mult'],
                 emb_path='../data/embeddings/char-embeddings.txt',
                 emb_dims=300):
    '''
    Load the embeddings and turn the datasets in a format that is
    compatible to the classifiers.
    '''
    embs, w2i = emb.load_embeddings(emb_path, emb_dims)

    clf = {}
    results = {}

    print('Original length of the datasets: train ({}), dev ({}), test ({})'.format(len(train), len(dev), len(test)))
    X_train, Y_train, train = load_dataset(train, combinations, embs, w2i)
    X_dev, Y_dev, dev = load_dataset(dev, combinations, embs, w2i)
    X_test, Y_test, test = load_dataset(test, combinations, embs, w2i)
    print('Length of the processed datasets (embeddings): train ({}), dev ({}), test ({})'.format(len(train), len(dev),
                                                                                                  len(test)))

    if len(train) < 20 or len(test) < 20:
        print('The dataset does not contain enough examples'.format(len(train), len(dev),
                                                                                                  len(test)))
        return {}

    for clf_name in clfs:
        clf[clf_name] = load_classifier(clf_name)

        for comb in X_train:

            if len(set(Y_test[comb])) == 1:
                print(
                    "It is useless to train a classifier to predict only one class: {}. Add classes to the data.".format(
                        set(Y_test[comb])))
                return {}

            clf[clf_name].fit(X_train[comb], Y_train[comb])
            print('\nClassifier: {}\nScore is: {}'.format(clf_name, clf[clf_name].score(X_test[comb], Y_test[comb])))
            print_predictions(Y_test[comb], clf[clf_name].predict(X_test[comb]), 10)
            results[clf_name + ' ' + comb] = (test, clf[clf_name].predict(X_test[comb]))
    return results


