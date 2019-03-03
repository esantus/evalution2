"""Generate metrics for a model."""

import datetime
import itertools
import os
import random

import numpy as np

from evalution import db, baseline_emb


def flatten_datasets(rel_datasets):
    """Take a dictionary of relations, and returns them in tuple format."""
    flattened_datasets = [[], [], []]
    for kind in rel_datasets.keys():
        for i in range(0, 3):
            for rel in rel_datasets[kind][i]:
                flattened_datasets[i].append([*rel, kind])
    return flattened_datasets


def get_random_non_rels(eval_db, number_of_words, relation):
    """Get a list of random word-pairs that aren't in a relation."""
    words = eval_db.all_words()
    not_rel = []
    while len(not_rel) < number_of_words:
        r1, r2 = random.sample(range(len(words) - 1), 2)
        random_pair = [words[r1], words[r2]]
        if not eval_db.are_rel(*random_pair, relation) and random_pair not in \
                not_rel:
            not_rel.append(random_pair)
    return not_rel


def split_relations(relations_to_test, splits=(60, 20, 20), size=1000,
                    flatten=True):
    """Returns a tuple containing training, validation and test dataset.

    Args:
        relations_to_test: list of relations to consider (e.g. [Relations['Synonym'], Relations['IsA']])
        splits: a triplet indicating the size of training, validation and test dataset
        size: size of the testet for each relation

    Returns:
        A named tuple with three Relations dictionaries (datasets): 'training', 'validation' and 'test'.
    """

    dataset_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '../data/dataset/'))
    db_path = os.path.join(dataset_path, 'evalution2.db')
    datasets = {k: [[], [], []] for k in relations_to_test}
    eval_db = db.EvaldDB(db_path, verbose=False)
    for rel_kind in relations_to_test:
        not_relations = get_random_non_rels(eval_db, size, rel_kind)
        if rel_kind != 'synonym':
            relations = eval_db.rel_pairs(rel_kind)
        else:
            relations = eval_db.all_synsets()
        random.shuffle(relations)
        # print(relations)
        for dataset_no in range(3):
            len_dataset = int(size / 100 * splits[dataset_no])
            for synset_relation in relations:
                word_relations = []
                if rel_kind != 'synonym':
                    words_in_domain = eval_db.words_in_synset(
                        synset_relation[0])
                    words_in_codomain = eval_db.words_in_synset(
                        synset_relation[1])
                    for w1, w2 in itertools.product(words_in_domain,
                                                    words_in_codomain):
                        word_relations.append((w1, w2, rel_kind))
                else:
                    words_in_synset = eval_db.words_in_synset(synset_relation)
                    for w1, w2 in itertools.combinations(words_in_synset, 2):
                        word_relations.append((w1, w2, rel_kind))
                if len(word_relations) * 2 < len_dataset:
                    non_relations_to_add = len(word_relations)
                else:
                    num_of_processed_relations = len(
                        datasets[rel_kind][dataset_no])
                    half_relations_left = int((len_dataset -
                                               num_of_processed_relations) / 2)
                    random.shuffle(word_relations)
                    word_relations = word_relations[:half_relations_left]
                    non_relations_to_add = half_relations_left
                for i in range(non_relations_to_add):
                    not_relation = not_relations.pop(i)
                    not_relation.append('not_' + rel_kind)
                    datasets[rel_kind][dataset_no].append(not_relation)
                num_of_processed_relations = len(datasets[rel_kind][dataset_no])
                if num_of_processed_relations + len(
                        word_relations) < len_dataset:
                    for r in word_relations:
                        datasets[rel_kind][dataset_no].append(r)
                else:
                    relations_left = len_dataset - num_of_processed_relations
                    # Too many relations.
                    if relations_left < 0:
                        datasets[rel_kind][dataset_no] = datasets[rel_kind][
                                                             dataset_no][
                                                         :relations_left]
                    else:
                        for _ in range(relations_left):
                            datasets[rel_kind][dataset_no].append(
                                word_relations.pop(
                                    random.randint(0, len(word_relations) - 1)))
                    break
    return datasets


def evaluate_model(true_relations, pred_relations):
    """Returns np y_true, y_pred arrays given a relation dictionary.

    Args:
        true_relations: list of relations in the test set
        pred_relations: list of predicted relations
    Returns:
        two numpy (sparse) arrays, one with the predicted values and another with the true values.
    TODO:
        Add support for binary classification (good if any relation exists)
    """
    # check for w1, r, w2 format
    # Evaluate one relation at the time.
    for rel_kind in pred_relations:
        all_relations = {rel for rel in
                         [true_relations[rel_kind], pred_relations[rel_kind]]}
        y_pred = [[] for _ in range(len(all_relations))]
        y_true = [[] for _ in range(len(all_relations))]
        for i, rel in enumerate(all_relations):
            if rel in true_relations[rel_kind]:
                y_true[i].append(1)
            else:
                y_true[i].append(0)

            if rel in pred_relations[rel_kind]:
                y_pred[i].append(1)
            else:
                y_pred[i].append(0)
    return np.array(y_true), np.array(y_pred)


def generate_report_data(y_true, y_pred, dataset=None, duration_run=None):
    """Generate a dictionary containing data to be used in the report.

    Args:
        dataset: bi-tuple containing triplets (W1, R, W2) used for training and validation
        duration_run: float indicating time taken for training.

    Returns:
        A dictionary with several evaluation data points that can be used with generate_report
    """

    metrics = {}
    now = datetime.datetime.now()
    metrics['date'] = now.strftime("%Y-%m-%d")
    metrics['time'] = now.strftime("%H:%M:%S")
    pred_classes = set(rel[1] for rel in y_pred)
    metrics['classes_no'] = len(pred_classes)
    metrics['pred_classes'] = '\n'.join(pred_classes)
    metrics['folds_no'] = 0
    metrics['run_duration'] = duration_run if duration_run else 'N/A'
    if dataset:
        metrics['training_no'] = len(dataset.training)
        metrics['training_pairs'] = '\n'.join(dataset.training)
        metrics['validation_no'] = len(dataset.validation)
        metrics['validation_pairs'] = '\n'.join(dataset.validation)
    return metrics


def format_report(report_data, pdf=False):
    """Pretty print a report from a report dictionary."""
    with open('report.md') as rep_file:
        report_raw = rep_file.read()
        report_formatted = report_raw.format(**report_data).strip()
        if pdf:
            # TODO: generate pdf from .md
            path = ''
            return path
        return report_formatted


def main():
    relations_to_test = ['synonym']
    datasets = split_relations(relations_to_test, splits=(80, 0, 20),
                               size=10000)
    results = baseline_emb.classify(*datasets['synonym'])
    pass
    # y_true, y_pred = evaluate_model(datasets[2], baseline.test_baseline())
    # report_data = generate_report_data(y_true, y_pred)
    # print(format_report(report_data))
    # print('\n----\nPDF generated at:' + format_report(report_data, pdf=True))


if __name__ == "__main__":
    main()
