from profiler.main import StepTaskResult
from profiler.cluster.methods.click import cluster as cluster_click
from profiler.cluster.methods.long_point import cluster as cluster_long_point
from profiler.cluster.methods.type import cluster as cluster_type
from profiler.cluster.methods.scroll import cluster as cluster_scroll
from profiler.cluster.methods.press import cluster as cluster_press
from profiler.cluster.methods.stop import cluster as cluster_stop
from profiler.cluster.methods.open import cluster as cluster_open
from profiler.cluster.methods.wait import cluster as cluster_wait


def cluster_result(result: StepTaskResult, *,
                   eps: int = 70,
                   ed_loose: float = 0.3,
                   ed_tight: float = 0.1):
    clusters = {}
    clusters.update(cluster_click(result, eps=eps))
    clusters.update(cluster_long_point(result, eps=eps))
    clusters.update(cluster_type(result, ed_loose=ed_loose, ed_tight=ed_tight))
    clusters.update(cluster_scroll(result))
    clusters.update(cluster_press(result))
    clusters.update(cluster_stop(result))
    clusters.update(cluster_open(result, ed_loose=ed_loose, ed_tight=ed_tight))
    clusters.update(cluster_wait(result))

    result.cluster = clusters
    return result
