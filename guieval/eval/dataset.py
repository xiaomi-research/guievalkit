import json
import os


class EvalDataset(object):

    def __init__(self, data_dir, subset, split="test"):
        self.data_dir = os.path.join(data_dir, split)
        self.subset = subset
        self.episode_data = self._load_data_()
        self.data = self._split_to_steps_(self.episode_data)

    def _load_data_(self):

        episode_paths = []
        for subset in self.subset:
            subdata_dir = os.path.join(self.data_dir, subset)
            if os.path.exists(subdata_dir):
                sequence_names = os.listdir(subdata_dir)
                for seq_name in sequence_names:
                    seq_dir = os.path.join(subdata_dir, seq_name)
                    if not os.path.isdir(seq_dir):
                        continue
                    episode_path = os.path.join(seq_dir, f"{seq_name}.json")
                    episode_paths.append(episode_path)

        ep_data = []
        for episode_path in episode_paths:
            try:
                with open(episode_path, "r") as f:
                    episode_data = json.load(f)
                    ep_data.append(episode_data)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed, file: {episode_path}, error: {e}")
            except Exception as e:
                print(f"Error occurred when loading file {episode_path}: {e}")
        return ep_data

    def _split_to_steps_(self, episode_data):
        data = []
        for edx, episode in enumerate(episode_data):
            for idx, step in enumerate(episode):
                try:
                    if step.get('subset') is None:
                        step['subset'] = step['image_path'].split('/')[0]
                    step['image_full_path'] = os.path.join(self.data_dir, step['image_path'])
                    data.append(step)
                except KeyError as e:
                    print(f"Missing key {e}, at episode {edx}, step {idx}")
                except Exception as e:
                    print(f"Error processing episode {edx}, step {idx}: {e}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
