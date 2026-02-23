from profiler.main import StepTaskResult  # internal


def get_stops(result: StepTaskResult):
    stop_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'STOP']

    stop_samples = [result_sample['prediction'].get('STATUS', '') for result_sample in stop_samples]

    return stop_samples


def cluster(result: StepTaskResult):
    stop_samples = get_stops(result)
    clusters = {'stop': stop_samples} if stop_samples else {}
    for cluster_label, cluster_stop_samples in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_stop_samples)
        result.clustered_decisions[cluster_label] = {
            "STATUS": cluster_stop_samples[0]
        }
    return clusters
