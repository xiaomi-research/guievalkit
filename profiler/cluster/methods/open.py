import random
from Levenshtein import ratio
from collections import defaultdict

# subsec internal
from profiler.main import StepTaskResult


def get_open_apps(result: StepTaskResult):
    open_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'OPEN']

    app_names = [result_sample['prediction'].get('OPEN_APP') for result_sample in open_samples]

    app_name_validated = []
    for app_name in app_names:
        if not isinstance(app_name, str):
            app_name_validated.append('')
        else:
            app_name_validated.append(app_name)

    return app_name_validated


def cluster(result: StepTaskResult, *,
            ed_loose: float = 0.3,
            ed_tight: float = 0.1):
    '''
    return {label: predictions}
    '''
    app_names = get_open_apps(result)

    if len(app_names) <= 1:
        return {f'open_{i}': [app_name] for i, app_name in enumerate(app_names)}

    clusters = defaultdict(list)

    clusters['open_0'].append(app_names[0])

    for app_name in app_names[1:]:
        similarities = [ratio(app_name, cluster_contents[0]) for cluster_contents in clusters.values()]

        for similarity, cluster_contents in zip(similarities, clusters.values()):
            if (app_name.lower() in cluster_contents[0].lower() or
                cluster_contents[0].lower() in app_name.lower()) and similarity > 1 - ed_loose:
                cluster_contents.append(app_name)
                break
        else:
            max_similarity = max(similarities)
            max_similarity_index = similarities.index(max_similarity)
            if max_similarity < 1 - ed_tight:
                clusters[f'open_{len(clusters)}'].append(app_name)
            else:
                clusters[f'open_{max_similarity_index}'].append(app_name)

    for cluster_label, cluster_app_names in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_app_names)
        result.clustered_decisions[cluster_label] = {
            "OPEN_APP": random.choice(cluster_app_names)
        }

    return clusters
