"""Generate metrics for a model."""

import collections
import datetime
import os
import pickle
import random

import numpy as np

from . import baseline


def split_relations(relations, splits=(60, 20, 20)):
    """Returns a tuple containing training, validation and test dataset.

    Args:
        relations: list of relations in Relations to consider (e.g. [Relations['Synonym'], Relations['IsA']])
        splits: a triplet indicating the size of training, validation and test dataset.

    Returns:
        A named tuple with three Relations dictionaries (datasets): 'training', 'validation' and 'test'.
    """

    Datasets = collections.namedtuple('Datasets', ['training', 'validation', 'test'])
    datasets = Datasets()
    for dataset in range(3):
        for rel_kind in relations:
            # TODO: does 60, 20, 20 means 60% of the total or 60% from each relation?
            dataset_instances = len(rel_kind) // splits[dataset]
            for _ in range(dataset_instances):
                rel_to_add = relations[rel_kind].pop(random.randint(0, len(relations[rel_kind])))
                datasets[dataset][rel_kind].append(rel_to_add)
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
        all_relations = {rel for rel in [true_relations[rel_kind], pred_relations[rel_kind]]}
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
    data_dir = ''
    relations = pickle.load(os.path.join(data_dir, 'all_relations_en.p'))
    splitted_relations = split_relations(relations, splits=(50, 30, 20))
    y_true, y_pred = evaluate_model(splitted_relations, baseline.test_baseline())
    report_data = generate_report_data(y_true, y_pred)
    print(format_report(report_data))
    print('\n----\nPDF generated at:' + format_report(report_data, pdf=True))


if __name__ == "__main__":
    main()