import random
from Levenshtein import ratio
from collections import defaultdict

# subsec internal
from profiler.main import StepTaskResult


def get_type_contents(result: StepTaskResult):
    type_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'TYPE']

    contents = [result_sample['prediction'].get('TYPE') for result_sample in type_samples]

    contents_validated = []
    for content in contents:
        if not isinstance(content, str):
            contents_validated.append('')
        else:
            contents_validated.append(content)

    return contents_validated


def cluster(result: StepTaskResult, *,
            ed_loose: float = 0.3,
            ed_tight: float = 0.1):
    '''
    return {label: predictions}
    '''
    contents = get_type_contents(result)

    if len(contents) <= 1:
        return {f'type_{i}': [content] for i, content in enumerate(contents)}

    clusters = defaultdict(list)

    clusters['type_0'].append(contents[0])

    for content in contents[1:]:
        similarities = [ratio(content, cluster_contents[0]) for cluster_contents in clusters.values()]

        for similarity, cluster_contents in zip(similarities, clusters.values()):
            if (content.lower() in cluster_contents[0].lower() or
                cluster_contents[0].lower() in content.lower()) and similarity > 1 - ed_loose:
                cluster_contents.append(content)
                break
        else:
            max_similarity = max(similarities)
            max_similarity_index = similarities.index(max_similarity)
            if max_similarity < 1 - ed_tight:
                clusters[f'type_{len(clusters)}'].append(content)
            else:
                clusters[f'type_{max_similarity_index}'].append(content)

    for cluster_label, cluster_contents in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_contents)
        result.clustered_decisions[cluster_label] = {
            "TYPE": random.choice(cluster_contents)
        }

    return clusters
