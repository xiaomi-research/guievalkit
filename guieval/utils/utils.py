import json
import os
import subprocess


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
        print(f"Failed to load {os.path.join(dataset_dir, config)}: {e}")
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
