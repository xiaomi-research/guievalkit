from collections import Counter

# subsec internal
from profiler.main import StepTaskResult


def get_scroll_directions(result: StepTaskResult):
    scroll_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'SCROLL']
    directions = [result_sample['prediction'].get('to') for result_sample in scroll_samples]

    directions_validated = []

    for direction in directions:
        if direction not in {"down", "up", "right", "left"}:
            directions_validated.append('unknown_direction')
        else:
            directions_validated.append(direction)

    return directions_validated


def cluster(result: StepTaskResult):
    directions = get_scroll_directions(result)

    clusters = Counter(directions)
    clusters = clusters.most_common()
    clusters = {f'scroll_{i}': [direction for _ in range(count)] for i, (direction, count) in enumerate(clusters)}

    for cluster_label, cluster_directions in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_directions)
        result.clustered_decisions[cluster_label] = {
            "POINT": (500, 500),
            "to": cluster_directions[0]
        }

    return clusters
