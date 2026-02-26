import numpy as np

from collections import defaultdict


def cal_result_entropy(cluster: dict[str, list]):
    distribution = np.array([len(cluster_contents) for cluster_contents in cluster.values()])
    distribution = distribution / distribution.sum()
    entropy = -np.sum(distribution * np.log2(distribution))
    return entropy


def cal_result_entropy_under_actions(clusters: dict[str, list]):
    actions = ['click', 'long_point', 'type', 'scroll', 'press', 'stop', 'open', 'wait']

    clusters_under_actions = defaultdict(dict)
    for decison, cluster_contents in clusters.items():
        for action in actions:
            if action in decison:
                clusters_under_actions[action][decison] = cluster_contents

    entropies = defaultdict(float)
    for action, action_clusters in clusters_under_actions.items():
        entropies[action] = cal_result_entropy(action_clusters)

    return entropies
