import json
import os
import subprocess
import itertools
import traceback
import sys
from PIL import Image
from qwen_vl_utils import fetch_image
from typing import Generator, Iterable

import logging

logger = logging.getLogger(__name__)


def str_default_none(value: str | None) -> str | None:
    if not value:
        return None
    elif isinstance(value, str) and value.lower() in ['none', 'null']:
        return None
    else:
        return str(value)


def get_gpu_list():
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if CUDA_VISIBLE_DEVICES != '':
        gpu_list = [int(x) for x in CUDA_VISIBLE_DEVICES.split(',')]
        return gpu_list
    try:
        ps = subprocess.Popen(('nvidia-smi', '--list-gpus'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
        return list(range(int(output)))
    except Exception:
        return []


def load_info(dataset_dir, config):
    try:
        with open(os.path.join(dataset_dir, config), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {os.path.join(dataset_dir, config)}: {e}")
        return {}


def load_json_data(file_path):
    data = []
    if file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            data = json.load(file)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
            try:
                json.loads(first_line)
                data.append(json.loads(first_line))
            except json.JSONDecodeError:
                pass
            for line in file:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def batched(iterable: Iterable, n: int) -> Generator[tuple, None, None]:
    '''
    Batch data into tuples of length n. The last batch may be shorter.
    '''
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def get_simplified_traceback(max_frames: int = 10) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_list = traceback.extract_tb(exc_traceback)
    limited_tb = tb_list[-max_frames:]
    formatted = traceback.format_list(limited_tb)
    exception_only = traceback.format_exception_only(exc_type, exc_value)

    return ''.join(formatted + exception_only)


def qwen_fetch_image(image_abspath: str, *,
                     min_pixels: int | None = None,
                     max_pixels: int | None = None,
                     patch_size: int | None = None) -> Image.Image:
    '''
    qwen_vl_utils.process_vision_info and this tool shared qwen_vl_utils.fetch_image
    with the same default value `image_patch_size=14`.

    If have used `qwen_vl_utils.process_vision_info` before to process the messages,
    we recommend using this function for aligned image processing to avoid inconsistency.

    Still, if you want to simply fetch the original image, just set `min_pixels` and `max_pixels` to None.

    Args:
        image_abspath: The absolute path of the image to fetch.
        min_pixels: The minimum number of pixels the image should have.
        max_pixels: The maximum number of pixels the image should have.

    Returns:
        The fetched image.
    '''
    if min_pixels is None and max_pixels is None:
        return Image.open(image_abspath)
    elif min_pixels is None or max_pixels is None:
        raise ValueError('min_pixels and max_pixels must be set together')
    else:
        image_info = dict(image_url=image_abspath)
        if min_pixels:
            image_info['min_pixels'] = min_pixels
        if max_pixels:
            image_info['max_pixels'] = max_pixels
        return (fetch_image(image_info) if patch_size is None else
                fetch_image(image_info, image_patch_size=patch_size))
