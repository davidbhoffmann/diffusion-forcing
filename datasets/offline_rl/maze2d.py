from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from maze_dataset import MazeDataset, MazeDatasetConfig
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import urllib
import os


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def download_dataset_from_url(save_dir, dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(save_dir, dataset_name)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class Maze2dOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.dataset_url = cfg.dataset_url
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        self.gamma = cfg.gamma
        self.n_frames = cfg.episode_len + 1
        self.total_steps = len(self.dataset["observations"])
        self.dataset["values"] = (
            self.compute_value(self.dataset["rewards"]) * (1 - self.gamma) * 4 - 1
        )

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            value[i] += self.gamma * value[i + 1]
        return value

    def __len__(self):
        return self.total_steps - self.n_frames + 1

    def __getitem__(self, idx):
        observation = torch.from_numpy(
            self.dataset["observations"][idx : idx + self.n_frames]
        ).float()
        action = torch.from_numpy(
            self.dataset["actions"][idx : idx + self.n_frames]
        ).float()
        reward = torch.from_numpy(
            self.dataset["rewards"][idx : idx + self.n_frames]
        ).float()
        # value = torch.from_numpy(self.dataset["values"][idx : idx + self.n_frames]).float()

        done = np.zeros(self.n_frames, dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)

        # goal = torch.zeros((self.n_frames, 0))

        return observation, action, reward, nonterminal

    def get_dataset(self):
        h5path = download_dataset_from_url(self.save_dir, self.dataset_url)
        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in get_keys(dataset_file):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        N_samples = data_dict["observations"].shape[0]

        if data_dict["rewards"].shape == (N_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]

        if data_dict["terminals"].shape == (N_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]

        return data_dict


class MultiMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.n_mazes = cfg.n_mazes
        self.grid_size = cfg.grid_size
        self.start_goal_pairs_per_maze = int(
            getattr(cfg, "start_goal_pairs_per_maze", 1)
        )
        os.makedirs(cfg.save_dir, exist_ok=True)

        expected_obs_dim = int((self.grid_size * 2 + 1) ** 2)
        split_path = os.path.join(self.save_dir, f"{split}.npz")
        needs_regen = not os.path.exists(split_path)
        if not needs_regen:
            split_data = np.load(split_path)
            observations = split_data["observations"]
            needs_regen = (
                observations.ndim != 2 or observations.shape[-1] != expected_obs_dim
            )
        if needs_regen:
            self.generate_data()
        self.dataset = dict(np.load(split_path))
        # Backward-compat: older generated files stored scalar actions as shape (N,)
        if self.dataset["actions"].ndim == 1:
            self.dataset["actions"] = self.dataset["actions"][:, None]

        if not bool(getattr(self.cfg, "_runtime_stats_initialized", False)):
            self._update_cfg_stats_from_dataset()

        self.gamma = cfg.gamma
        self.n_frames = cfg.episode_len + 1
        self.total_steps = len(self.dataset["observations"])
        self.dataset["values"] = (
            self.compute_value(self.dataset["rewards"]) * (1 - self.gamma) * 4 - 1
        )

    def _update_cfg_stats_from_dataset(self):
        observations = self.dataset["observations"]
        actions = self.dataset["actions"]
        rewards = self.dataset["rewards"]

        if observations.size == 0:
            return

        max_stats_samples = int(getattr(self.cfg, "max_stats_samples", 200000))
        n = observations.shape[0]
        if n > max_stats_samples:
            idx = np.random.default_rng(0).choice(n, size=max_stats_samples, replace=False)
            observations = observations[idx]
            actions = actions[idx]
            rewards = rewards[idx]

        observation_mean = observations.mean(axis=0)
        observation_std = observations.std(axis=0)
        observation_std = np.where(np.abs(observation_std) < 1e-6, 1.0, observation_std)

        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        action_std = np.where(np.abs(action_std) < 1e-6, 1.0, action_std)

        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        if abs(reward_std) < 1e-6:
            reward_std = 1.0

        with open_dict(self.cfg):
            self.cfg.observation_shape = [int(observations.shape[-1])]
            self.cfg.observation_mean = observation_mean.tolist()
            self.cfg.observation_std = observation_std.tolist()
            self.cfg.action_mean = action_mean.tolist()
            self.cfg.action_std = action_std.tolist()
            self.cfg.reward_mean = reward_mean
            self.cfg.reward_std = reward_std
            self.cfg._runtime_stats_initialized = True

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            value[i] += self.gamma * value[i + 1]
        return value

    def __len__(self):
        return self.total_steps - self.n_frames + 1

    def __getitem__(self, idx):
        observation = torch.from_numpy(
            self.dataset["observations"][idx : idx + self.n_frames]
        ).float()
        action = torch.from_numpy(
            self.dataset["actions"][idx : idx + self.n_frames]
        ).float()
        reward = torch.from_numpy(
            self.dataset["rewards"][idx : idx + self.n_frames]
        ).float()

        done = np.zeros(self.n_frames, dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)

        return observation, action, reward, nonterminal

    def _shortest_path(self, wall_mask, start, goal):
        if start == goal:
            return [start]

        h, w = wall_mask.shape
        q = deque([start])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if wall_mask[nr, nc]:
                    continue
                nxt = (nr, nc)
                if nxt in parent:
                    continue
                parent[nxt] = (r, c)
                if nxt == goal:
                    path = [goal]
                    while path[-1] is not None:
                        prev = parent[path[-1]]
                        if prev is None:
                            break
                        path.append(prev)
                    return list(reversed(path))
                q.append(nxt)

        return None

    def _sample_start_goal_pairs(self, free_cells, n_pairs, rng):
        pairs = []
        seen = set()
        if len(free_cells) < 2:
            return pairs

        max_attempts = max(100, n_pairs * 30)
        attempts = 0
        while len(pairs) < n_pairs and attempts < max_attempts:
            s_idx, g_idx = rng.choice(len(free_cells), size=2, replace=False)
            start = tuple(free_cells[s_idx])
            goal = tuple(free_cells[g_idx])
            key = (start, goal)
            attempts += 1
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        return pairs

    def _encode_single_channel_observation(self, wall_mask, curr, goal):
        obs = np.zeros(wall_mask.shape, dtype=np.float32)
        obs[wall_mask] = 1.0
        obs[goal[0], goal[1]] = 3.0
        if curr == goal:
            obs[curr[0], curr[1]] = 4.0
        else:
            obs[curr[0], curr[1]] = 2.0
        return obs.flatten()

    def generate_data(self, train_frac=0.8, val_frac=0.1):
        cfg = MazeDatasetConfig(
            name="base", grid_n=self.grid_size, n_mazes=self.n_mazes
        )
        mazes = list(MazeDataset.from_config(cfg))
        np.random.shuffle(mazes)

        n = self.n_mazes
        t_idx = int(n * train_frac)
        v_idx = int(n * (train_frac + val_frac))

        splits = {
            "training": mazes[:t_idx],
            "validation": mazes[t_idx:v_idx],
            "test": mazes[v_idx:],
        }

        results = {}
        rng = np.random.default_rng(0)
        for split, split_mazes in splits.items():
            obs, acts, rews = [], [], []

            for maze in split_mazes:
                pixels = maze.as_pixels(False, False)
                wall_mask = pixels[:, :, 2] >= 254
                free_cells = np.argwhere(~wall_mask)
                pairs = self._sample_start_goal_pairs(
                    free_cells, self.start_goal_pairs_per_maze, rng
                )

                for start, goal in pairs:
                    path_pixels = self._shortest_path(wall_mask, start, goal)
                    if not path_pixels:
                        continue

                    for i, curr in enumerate(path_pixels):
                        if i < len(path_pixels) - 1:
                            nxt = path_pixels[i + 1]
                            if nxt[0] < curr[0]:
                                a = 0
                            elif nxt[0] > curr[0]:
                                a = 1
                            elif nxt[1] < curr[1]:
                                a = 2
                            else:
                                a = 3
                            r = 0.0
                        else:
                            a, r = 0, 1.0

                        o = self._encode_single_channel_observation(
                            wall_mask, tuple(curr), tuple(goal)
                        )

                        obs.append(o)
                        acts.append([a])
                        rews.append(r)

            results[split] = {
                "observations": np.stack(obs) if obs else np.array([]),
                "actions": np.array(acts),
                "rewards": np.array(rews),
            }
            save_path = os.path.join(self.save_dir, f"{split}.npz")
            np.savez(save_path, **results[split])

        return results


if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gym

    os.chdir("../..")
    cfg = MagicMock()
    cfg.env_id = "maze2d-medium-v1"
    cfg.episode_len = 600
    cfg.gamma = 1.0
    dataset = Maze2dOfflineRLDataset(cfg)
    o, a, r, n = dataset.__getitem__(10)
    print(o.shape, a.shape, r.shape, n.shape)
    plt.figure()
    plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")

    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]

    maze_string = gym.make(cfg.env_id).str_maze_spec
    grid = convert_maze_string_to_grid(maze_string)

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle(
                    (i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black"
                )
                plt.gca().add_patch(square)

    start_x, start_y = o[..., 0, :2]
    start_circle = plt.Circle(
        (start_x, start_y), 0.16, facecolor="white", edgecolor="black"
    )
    plt.gca().add_patch(start_circle)
    inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    plt.gca().add_patch(inner_circle)

    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (
            2 * num_points
        )
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        plt.gca().add_patch(star)

    goal_x, goal_y = o[..., -1, :2]
    goal_circle = plt.Circle(
        (goal_x, goal_y), 0.16, facecolor="white", edgecolor="black"
    )
    plt.gca().add_patch(goal_circle)
    draw_star((goal_x, goal_y), radius=0.08)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("lightgray")
    plt.gca().set_axisbelow(True)
    plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    plt.xlim([0.5, len(grid) + 0.5])
    plt.ylim([0.5, len(grid[0]) + 0.5])
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    plt.grid(True, color="white", which="minor", linewidth=4)
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.show()
    print("Done.")
