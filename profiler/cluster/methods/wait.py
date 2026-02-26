from profiler.main import StepTaskResult  # internal


def get_waits(result: StepTaskResult):
    wait_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'WAIT']

    wait_samples = [result_sample['prediction'].get('duration', '') for result_sample in wait_samples]

    return wait_samples


def cluster(result: StepTaskResult):
    wait_samples = get_waits(result)
    clusters = {'wait': wait_samples} if wait_samples else {}

    for cluster_label, cluster_wait_samples in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_wait_samples)
        result.clustered_decisions[cluster_label] = {
            "duration": cluster_wait_samples[0]
        }
    return clusters
