"""Graph interface library for semantic relations."""

import copy
import random

import numpy as np
import pandas as pd


class Edge:
    def __init__(self, v1, v2, weight=1):
        self.v1 = v1
        self.v2 = v2
        self.w = weight


class Graph:
    def __init__(self, edges=None):
        if edges is None:
            edges = []
        self.vertices = []
        self.edges = copy.deepcopy(edges)

        for edge in edges:
            self.vertices.append(edge.v1)
            self.vertices.append(edge.v2)

    def add_edge(self, v1, v2, w):
        edge = Edge(v1, v2, w)
        self.edges.append(copy.deepcopy(edge))
        self.vertices.append(edge.v1)
        self.vertices.append(edge.v2)


def create_graph(filename, separator=',', rel=None, weighted=False):
    """Create a graph from a csv file containing relations and annotations."""
    g = Graph()
    data = pd.read_csv(filename, separator)
    m, _ = data.shape
    if rel and not isinstance(rel, list) and not isinstance(rel, tuple):
        rel = [rel]

    edge_dict = {}
    for i in range(m):
        if rel is not None and data['relation'][i] in rel:
            continue

        w = 1
        if weighted:
            w = data['relation_weight'][i] < 0.3

        w1 = data['word1_id'][i]
        w2 = data['word1_id'][i]

        if (w1, w2) not in edge_dict:
            edge_dict[(w1, w2)] = {}
        edge_dict[(w1, w2)][data['relation'][i]] = w

    for edge in edge_dict:
        if rel:
            for r in rel:
                if rel not in edge:
                    w = -1
                else:
                    w = edge[r]
                g.add_edge(edge.v1, edge.v2, w)
    return g


def build_rnd_graph(golden, rel, seed=None):
    """Build a random graph for testing."""
    def add_word(word):
        if word not in words:
            words.add(word)

    def add_edge(rel, word1, word2):
        data.append((rel, word1, word2))

    random.seed(seed)
    m, _ = golden.shape

    words = set()
    for i in range(m):
        if golden['relation'][i] != rel:
            continue
        add_word(golden['word1_id'][i])
        add_word(golden['word2_id'][i])

    data = []

    for word1 in words:
        for word2 in words:
            if word1 >= word2:
                continue
            if random.randint(0, 1):
                add_edge(rel, word1, word2)
                add_edge(rel, word2, word1)

    df = pd.DataFrame(data, columns=('relation', 'word1_id', 'word2_id'),
                      index=range(len(data)))
    return df


def evaluate_graph(graphs, relations, golden):
    """Compare two set of graphs and return a y_true and y_pred array."""
    if not isinstance(graphs, list) and not isinstance(graphs, tuple):
        graphs = [graphs]
    else:
        graphs = graphs
    if not isinstance(relations, list) and not isinstance(relations, tuple):
        relations = [relations]
    else:
        relations = relations

    all_edges = set()
    user_edges = [set() for _ in range(len(graphs))]
    golden_edges = [set() for _ in range(len(graphs))]

    for idx in range(len(relations)):
        rel = relations[idx]
        data = graphs[idx]
        if rel not in list(golden['relation']):
            # in case relation not found in golden, we may think of user graph
            # as of graph that didn't match any edges of golden at all
            return np.zeros(data.shape[0]), np.ones(data.shape[0])

        for i in range(data.shape[0]):
            vertex_1 = data['word1_id'][i]
            vertex_2 = data['word2_id'][i]

            if vertex_1 > vertex_2:
                vertex_1, vertex_2 = vertex_2, vertex_1
            user_edges[idx].add((vertex_1, vertex_2))
            all_edges.add((vertex_1, vertex_2))

        for i in range(golden.shape[0]):
            if golden['relation'][i] != rel:
                continue
            vertex_1 = data['word1_id'][i]
            vertex_2 = data['word2_id'][i]
            if vertex_1 > vertex_2:
                vertex_1, vertex_2 = vertex_2, vertex_1
            golden_edges[idx].add((vertex_1, vertex_2))
            all_edges.add((vertex_1, vertex_2))

    y_pred = [[] for _ in range(len(graphs))]
    y_true = [[] for _ in range(len(graphs))]

    for edge in all_edges:
        for idx in range(len(relations)):
            if edge in golden_edges[idx]:
                y_true[idx].append(1)
            else:
                y_true[idx].append(0)

            if edge in user_edges[idx]:
                y_pred[idx].append(1)
            else:
                y_pred[idx].append(0)

    return np.array(y_true), np.array(y_pred)



