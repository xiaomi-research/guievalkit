from collections import Counter

# subsec internal
from profiler.main import StepTaskResult


def get_press_buttons(result: StepTaskResult):
    press_samples = [result_sample for result_sample in result.result_samples if result_sample['action'] == 'PRESS']
    buttons = [result_sample['prediction'].get('PRESS') for result_sample in press_samples]

    buttons_validated = []
    for button in buttons:
        if button not in {"BACK", "HOME", "ENTER"}:
            buttons_validated.append('unknown_button')
        else:
            buttons_validated.append(button)

    return buttons_validated


def cluster(result: StepTaskResult):
    buttons = get_press_buttons(result)
    clusters = Counter(buttons)
    clusters = clusters.most_common()
    clusters = {f'press_{i}': [button for _ in range(count)] for i, (button, count) in enumerate(clusters)}

    for cluster_label, cluster_buttons in clusters.items():
        result.clustered_distribution[cluster_label] = len(cluster_buttons)
        result.clustered_decisions[cluster_label] = {
            "PRESS": cluster_buttons[0]
        }

    return clusters
