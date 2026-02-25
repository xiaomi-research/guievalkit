import numpy as np
from sklearn.cluster import OPTICS
from typing import Sequence
from numbers import Number
from collections import defaultdict

# subsec internal
from profiler.main import StepTaskResult


def get_click_points(result: StepTaskResult):
    click_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'CLICK']

    coords = [result_sample['prediction'].get('POINT') for result_sample in click_samples]

    coords_validated = []
    for coord in coords:
        if not (isinstance(coord, Sequence) and len(coord) == 2 and all(isinstance(c, Number) for c in coord)):
            coords_validated.append(tuple([None, None]))
        else:
            coords_validated.append(tuple(coord))

    return coords_validated


def cluster(result: StepTaskResult, eps: float = 70):
    '''
    return {label: predictions}
    '''
    points = np.array(get_click_points(result), dtype=float)

    if points.shape[0] <= 1:
        return {f'click_{i}': [point.tolist()] for i, point in enumerate(points)}

    opt = OPTICS(min_samples=(min(2, len(points)) if len(points) <= 7 else 5),
                 cluster_method='dbscan',
                 eps=eps).fit(points)

    labels = opt.labels_

    isolated_dot_idx = np.where(labels == -1)[0]
    clusters = defaultdict(list)

    cluster_labels = set()
    for label, point in zip(labels, points):
        if label != -1:
            clusters[f'click_{label}'].append(point.tolist())
            cluster_labels.add(f'click_{label}')

    for i, isolated_dot in enumerate(isolated_dot_idx):
        clusters[f'click_isolated_{i}'].append(points[isolated_dot].tolist())
        cluster_labels.add(f'click_isolated_{i}')

    for cluster_label in cluster_labels:
        coordinates1, coordinates2 = zip(*clusters[cluster_label])
        coordinates1_mean = np.mean(coordinates1)
        coordinates2_mean = np.mean(coordinates2)
        result.clustered_decisions[cluster_label] = {
            "POINT": (int(coordinates1_mean), int(coordinates2_mean))
        }
        result.clustered_distribution[cluster_label] = len(clusters[cluster_label])

    return clusters
