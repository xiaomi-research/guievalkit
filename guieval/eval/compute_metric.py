import numpy as np

from collections import defaultdict


def compute_episode_metrics(step_results, no_open_action=False):

    episode_results = defaultdict(list)
    for result in step_results:
        subset = result.get("subset")
        episode_id = result.get("episode_id")
        if subset is None or episode_id is None:
            print(f"Result is missing subset or episode_id: {result}")
            continue
        if no_open_action and result.get("answer", {}).get("action_type") == "open":
            continue
        episode_key = f"{subset}-{episode_id}"
        episode_results[episode_key].append(result)

    success, progress = [], []
    total_exact_matches = 0
    for _, eplist in episode_results.items():
        ep_success, ep_progress = True, 0
        for ex in sorted(eplist, key=lambda x: x['step_id']):
            if ex['exact_match'] is True:
                ep_progress += 1
                total_exact_matches += 1
            else:
                ep_success = False
            if not ep_success:
                break
        success.append(ep_success)
        progress.append(ep_progress / len(eplist) * 1.0)

    total_steps = 0
    for _, eplist in episode_results.items():
        for _ in eplist:
            total_steps += 1

    num_episodes = len(success)
    num_successes = sum(success)
    return {
        "total_episodes": num_episodes,
        "total_steps": total_steps,
        "num_successes": num_successes,
        "total_exact_matches": total_exact_matches,
        "success_rate": round(sum(success) / len(success), 4),
        "goal_progress": round(sum(progress) / len(progress), 4)}


def compute_atomic_metrics(step_results):
    recorder = {
        'total': {'count': 0, 'type_match': 0, 'exact_match': 0, "hit": 0},
        'CLICK': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'TYPE': {'count': 0, 'type_match': 0, 'exact_match': 0, 'text_dist': []},
        'SCROLL': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'PRESS': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'STOP': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'LONG_POINT': {'count': 0, 'type_match': 0, 'exact_match': 0},
        'OPEN': {'count': 0, 'type_match': 0, 'exact_match': 0, 'text_dist': []},
        'WAIT': {'count': 0, 'type_match': 0, 'exact_match': 0},
    }
    for step in step_results:
        recorder['total']['count'] += 1
        recorder['total']['hit'] += step.get('format_hit', 0)

        action_type = step.get('answer', {}).get('action_type')
        if isinstance(action_type, str):
            action_type = action_type.upper()
        else:
            action_type = ''

        if action_type in recorder:
            recorder[action_type]['count'] += 1
            recorder[action_type]['type_match'] += step.get('type_match', 0)
            recorder['total']['type_match'] += step.get('type_match', 0)
            recorder[action_type]['exact_match'] += step.get('exact_match', 0)
            recorder['total']['exact_match'] += step.get('exact_match', 0)
            if 'text_dist' in recorder[action_type] and step.get('text_dist') is not None:
                recorder[action_type]['text_dist'].append(step['text_dist'])

    scores = {
        metric_key: {
            'count': recorder[metric_key]['count'],
            'type_acc': round(
                recorder[metric_key]['type_match'] / recorder[metric_key]['count'], 4
            ) if recorder[metric_key]['count'] > 0 else 0,
            'exact_acc': round(
                recorder[metric_key]['exact_match'] / recorder[metric_key]['count'], 4
            ) if recorder[metric_key]['count'] > 0 else 0
        }
        for metric_key in ['total', 'CLICK', 'LONG_POINT', 'SCROLL', 'PRESS', 'STOP', 'TYPE', 'OPEN', 'WAIT']
    }

    scores['total_no_open'] = {
        'count': recorder['total']['count'] - recorder['OPEN']['count'],
        'type_acc': round(
            (recorder['total']['type_match'] - recorder['OPEN']['type_match']) / (
                    recorder['total']['count'] - recorder['OPEN']['count']),
            4) if (recorder['total']['count'] - recorder['OPEN']['count']) > 0 else 0,
        'exact_acc': round(
            (recorder['total']['exact_match'] - recorder['OPEN']['exact_match']) / (
                    recorder['total']['count'] - recorder['OPEN']['count']),
            4) if (recorder['total']['count'] - recorder['OPEN']['count']) > 0 else 0
    }

    scores['total']['hit_rate'] = round(
        recorder['total']['hit'] / recorder['total']['count'], 4) if recorder['total']['count'] > 0 else 0

    if recorder['TYPE']['text_dist']:
        scores['TYPE']['text_dist_avg'] = round(
            sum(recorder['TYPE']['text_dist']) / len(recorder['TYPE']['text_dist']), 4)
    else:
        scores['TYPE']['text_dist_avg'] = 0

    if recorder['OPEN']['text_dist']:
        scores['OPEN']['text_dist_avg'] = round(
            sum(recorder['OPEN']['text_dist']) / len(recorder['OPEN']['text_dist']), 4)
    else:
        scores['OPEN']['text_dist_avg'] = 0

    pixel_distances = [step['pixel_distance'] for step in step_results if step.get('pixel_distance') is not None]

    median_pixel_distance = round(float(np.median(pixel_distances)), 4) if pixel_distances else -1

    mean_pixel_distance = -1

    if pixel_distances:
        pixel_distances = np.array(pixel_distances)
        filtered_distances = pixel_distances[pixel_distances < 1e15]
        if len(filtered_distances) > 0:
            mean_pixel_distance = round(float(np.mean(filtered_distances)), 4)

    scores['mean_pixel_distance'] = mean_pixel_distance
    scores['median_pixel_distance'] = median_pixel_distance

    return scores
