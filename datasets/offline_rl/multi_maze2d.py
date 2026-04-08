from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import deque
import itertools
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

from datasets.offline_rl.utils import get_solutions_tree, get_action



class MultiMaze2dOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.save_dir = cfg.save_dir
        self.n_mazes = cfg.n_mazes
        self.maze_size = cfg.maze_size
        self.gird_size = self.maze_size * 2 + 1
        self.gamma = cfg.gamma
        self.n_frames = cfg.episode_len + 1
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        self.n_trajectories = int(self.dataset["actions"].shape[0])
        self.set_dataset_stats()

        # REMOVE Backward-compat: older generated files stored scalar actions as shape (N,)
        # if self.dataset["actions"].ndim == 1:
        #     # TODO is this used?
        #     self.dataset["actions"] = self.dataset["actions"][:, None]

        # if not bool(getattr(self.cfg, "_runtime_stats_initialized", False)):
        #     # TODO update function
        #     self._update_cfg_stats_from_dataset()

        # self.total_steps = len(self.dataset["observations"])
        # self.valid_starts = self._build_valid_starts(
        #     self.dataset["terminals"], self.n_frames
        # )
        

        # # TODO move this inside the get_dataset method
        # self.dataset["values"] = (
        #     self.compute_value(self.dataset["rewards"], self.dataset["terminals"])
        #     * (1 - self.gamma)
        #     * 4
        #     - 1
        # )

    def compute_value(self, reward, terminals):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            if terminals[i]:
                continue
            value[i] += self.gamma * value[i + 1]
        return value

    # def _build_valid_starts(self, terminals, n_frames):
    #     assert len(terminals) > n_frames

    #     valid_starts = []
    #     max_start = len(terminals) - n_frames
    #     for start in range(max_start + 1):
    #         # Allow a terminal only at the last index of the sampled window.
    #         if np.any(terminals[start : start + n_frames - 1]):
    #             continue
    #         valid_starts.append(start)
    #     return np.array(valid_starts, dtype=np.int64)

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        # start = int(self.valid_starts[idx])
        # end = start + self.n_frames

        # observation = torch.from_numpy(
        #     self.dataset["observations"][start:end]
        # ).float()
        # action = torch.from_numpy(
        #     self.dataset["actions"][start:end]
        # ).float()
        # reward = torch.from_numpy(
        #     self.dataset["rewards"][start:end]
        # ).float()


        grid_pid, goal_pid = self.dataset["positions"][idx][0].tolist()
        positions = torch.from_numpy(self.dataset["positions"][idx][1:]).float()
        grid_aid, goal_aid = self.dataset["actions"][idx][:2].tolist()
        actions = torch.from_numpy(self.dataset["actions"][idx][2:]).float()
        grid_rid, goal_rid = self.dataset["rewards"][idx][:2].tolist()
        rewards = torch.from_numpy(self.dataset["rewards"][idx][2:]).float()


        # TODO This is more for debugging, remove later.
        if grid_pid!=grid_aid or goal_pid!=goal_aid:
            raise RuntimeError("Some bug with the maze and goal ids.")
        if len(actions) != len(positions):
            raise RuntimeError("Some bug with length of observaitons")
            
        goal = torch.from_numpy(self.dataset["goals"][goal_pid])
        maze = torch.from_numpy(self.dataset["grids"][grid_pid])

        # rewards = torch.zeros_like(positions).float()
        # rewards[-1] = 1.0
        nonterminals = torch.ones_like(positions).bool()
        nonterminals[-1] = False

        # return observation, action, reward, nonterminal
        return maze, goal, positions, actions, rewards, nonterminals

        
        


    # def _encode_single_channel_observation(self, wall_mask, curr, goal):
    #     obs = np.zeros(wall_mask.shape, dtype=np.float32)
    #     obs[wall_mask] = 1.0
    #     obs[goal[0], goal[1]] = 3.0
    #     if curr == goal:
    #         obs[curr[0], curr[1]] = 4.0
    #     else:
    #         obs[curr[0], curr[1]] = 2.0
    #     return obs.flatten()

    # def _lattice_nodes(self, wall_mask):
    #     side_x, side_y = wall_mask.shape
    #     nodes = []
    #     for x in range(1, side_x, 2):
    #         for y in range(1, side_y, 2):
    #             if not bool(wall_mask[x, y]):
    #                 nodes.append((x, y))
    #     return nodes

    # def _shortest_path(self, wall_mask, start, goal):
    #     if start == goal:
    #         return [start]
    #     q = deque([start])
    #     parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    #     while q:
    #         x, y = q.popleft()
    #         for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
    #             nx, ny = x + dx, y + dy
    #             if nx < 1 or ny < 1 or nx >= wall_mask.shape[0] or ny >= wall_mask.shape[1]:
    #                 continue
    #             midx, midy = x + dx // 2, y + dy // 2
    #             if bool(wall_mask[midx, midy]) or bool(wall_mask[nx, ny]):
    #                 continue
    #             nxt = (nx, ny)
    #             if nxt in parent:
    #                 continue
    #             parent[nxt] = (x, y)
    #             if nxt == goal:
    #                 q.clear()
    #                 break
    #             q.append(nxt)

    #     if goal not in parent:
    #         return None

    #     path = []
    #     cur = goal
    #     while cur is not None:
    #         path.append(cur)
    #         cur = parent[cur]
    #     path.reverse()
    #     return path

    def get_dataset(self):
        dataset_paths = {
            split: os.path.join(self.save_dir, f"{split}_mazes.npz")
            for split in ["training", "validation", "test"]
        }
        # Generate data if it doesn't exist
        if not os.path.exists(dataset_paths[self.split]):
            print("Generating Maze Dataset...")
            self.generate_data(dataset_paths)
            print("saved at", dataset_paths)
        # Load and prepare data
        with np.load(dataset_paths[self.split]) as raw:
            dataset = {k: raw[k] for k in raw.files}

        if dataset["actions"].shape[-1] - 2  != self.n_frames:
            raise RuntimeError(
                "Dataset shape does not match n_frames or action_dim specified in the config.\n"
                f"-> Actions shape: {dataset['actions'].shape}\n"
                f"-> Positions shape: {dataset['positions'].shape}\n"
                f"-> N frames: {self.n_frames}"
            )
        if dataset["grids"].shape[-1] != self.gird_size**2:
            raise RuntimeError(
                "Dataset grid size does't match the cfg.\n"
                f"-> Dataset grid shape: {dataset['grids'].shape}\n"
                f"-> Config grid shape: {(self.gird_size, self.gird_size)}"
            )
        if dataset["positions"].shape[-1] != self.cfg.observation_shape:
            raise RuntimeError(
                "Dataset observation size does't match the cfg.\n"
                f"-> Dataset observation shape: {dataset['positions'].shape}\n"
                f"-> Config grobservationid shape: {self.cfg.observation_shape}"
            )
        if dataset["goals"].shape[-1] != self.cfg.goal_dim:
            raise RuntimeError(
                "Dataset goal size does't match the cfg.\n"
                f"-> Dataset goal ervation shape: {dataset['goals'].shape}\n"
                f"-> Config goal shape: {self.cfg.goal_dim}"
            )
        return dataset
        

    def generate_data(self, dataset_paths, train_frac=0.8, val_frac=0.1):
        cfg = MazeDatasetConfig(
            name="base", grid_n=self.maze_size, n_mazes=self.n_mazes
        )
        # TODO currently it is possible that duplicates exist between the 
        # different data splits -> leakage. 
        dataset = MazeDataset.from_config(cfg).filter_by.path_length(min_length=self.n_frames)
        max_iter, i = 100, 0
        while len(dataset.mazes) < self.n_mazes and i < max_iter:
            batch = MazeDataset.from_config(cfg).filter_by.path_length(min_length=self.n_frames)
            dataset.mazes.extend(batch.mazes)
            i+=1
        mazes = dataset.mazes[:self.n_mazes]
        print(f"Generated {len(mazes)} mazes with solutions longer than {self.n_frames}.")
        np.random.shuffle(mazes)

        n = self.n_mazes
        t_idx = int(n * train_frac)
        v_idx = int(n * (train_frac + val_frac))

        splits = {
            "training": mazes[:t_idx],
            "validation": mazes[t_idx:v_idx],
            "test": mazes[v_idx:],
        }
        # Start positions are the same for any maze of the same size.
        # Still requrie filtering by path to goal length
        starts = list(itertools.product(range(self.maze_size), repeat=2))
        for split, split_mazes in splits.items():
            # obs, acts, rews, terminals, grid_ids, trajectory_ids = [], [], [], [], [], []
            grids, goals, actions, positions, rewards = [], [], [], [], []
            grid_id, goal_id = 0, 0 
            for grid_id, maze in enumerate(split_mazes):
                pixels = maze.as_pixels(False, False)
                wall_mask = pixels[..., 0] == 255
                wall_mask = wall_mask.reshape(-1)
                goal = tuple(maze.end_pos)
                grids.append(wall_mask.astype(np.uint8))
                goals.append(goal) 
                # TODO: check if fixing trajectory length leads to problems
                # Could also take trajectories longer than n_frames and take an n_frame 
                # slice out of them. Sequences wouldn't always end in goal.
                paths_tree, start_nodes = get_solutions_tree(maze, goal, self.n_frames-1)
                for start_node in start_nodes:
                    curr = tuple(start_node)
                    traj_position, traj_actions = [], []
                    while curr:
                        traj_position.append(curr)
                        nxt = paths_tree[curr]
                        traj_actions.append(get_action(curr, nxt))
                        curr = nxt
                    # Each trajectory is specific to a maze and a goal 
                    traj_rewards = [0]*len(traj_actions)
                    traj_rewards[-1] = 1
                    rewards.append([grid_id, goal_id] + traj_rewards)
                    actions.append([grid_id, goal_id] + traj_actions) 
                    positions.append([(grid_id, goal_id)] + traj_position) 
                

            np.savez(
                dataset_paths[split],
                grids=np.array(grids, dtype=np.uint8),
                goals=np.array(goals),
                actions=np.array(actions),  
                rewards=np.array(rewards),
                positions=np.array(positions),
            )


                # All possible start positions
                

                # np.random.shuffle(starts)

                # # n_sampled = 0
                # for start in starts:
                #     if start == goal:
                #         continue
                #     path = self._shortest_path(wall_mask, start, goal)
                #     if path is None or len(path) < self.n_frames:
                #         continue

                #     for i, curr in enumerate(path):
                #         is_terminal = i == len(path) - 1
                #         if i < len(path) - 1:
                #             nxt = path[i + 1]
                #             if nxt[0] < curr[0]:
                #                 a = 1  # Up
                #             elif nxt[0] > curr[0]:
                #                 a = 2  # Down
                #             elif nxt[1] < curr[1]:
                #                 a = 3  # Left
                #             else:
                #                 a = 4  # Right
                #             r = 0.0
                #         else:
                #             # Static-goal reward map: zero everywhere except terminal goal state.
                #             a, r = 0, 1.0

                #         o = self._encode_single_channel_observation(
                #             wall_mask, tuple(curr), goal
                #         )

                #         obs.append(o)
                #         acts.append([a])
                #         rews.append(r)
                #         terminals.append(is_terminal)
                #         grid_ids.append(grid_id)
                #         trajectory_ids.append(traj_id)

                #     # n_sampled += 1
                #     traj_id += 1

                # if n_sampled == 0:
                #     solution_path = [tuple(x) for x in maze.solution]
                #     if len(solution_path) >= self.n_frames:
                #         for i, curr in enumerate(solution_path):
                #             is_terminal = i == len(solution_path) - 1
                #             if i < len(solution_path) - 1:
                #                 nxt = solution_path[i + 1]
                #                 if nxt[0] < curr[0]:
                #                     a = 1
                #                 elif nxt[0] > curr[0]:
                #                     a = 2
                #                 elif nxt[1] < curr[1]:
                #                     a = 3
                #                 else:
                #                     a = 4
                #                 r = 0.0
                #             else:
                #                 a, r = 0, 1.0

                #             o = self._encode_single_channel_observation(
                #                 wall_mask, curr, goal
                #             )
                #             obs.append(o)
                #             acts.append([a])
                #             rews.append(r)
                #             terminals.append(is_terminal)
                #             grid_ids.append(grid_id)
                #             trajectory_ids.append(traj_id)
                #         traj_id += 1

            # if len(obs) == 0:
            #     raise RuntimeError(f"No trajectories were generated for split '{split}'.")
            
            # np.savez(
            #     dataset_paths[split],
            #     grids=np.stack(grids),
            #     grid_ids=np.array(grid_ids, dtype=np.int32),
            #     trajectory_ids=np.array(trajectory_ids, dtype=np.int32),
            #     observations=np.stack(obs),
            #     actions=np.array(acts),
            #     rewards=np.array(rews),
            #     terminals=np.array(terminals, dtype=bool),
            # )

    def set_dataset_stats(self):
        grids = self.dataset["grids"]
        goals = self.dataset["goals"]
        actions = self.dataset["actions"]
        positions = self.dataset["positions"]
        rewards = self.dataset["rewards"]


        max_stats_samples =  200000
        n = actions.shape[0]
        if n > max_stats_samples:
            idx = np.random.default_rng(0).choice(
                n, size=max_stats_samples, replace=False
            )
            positions = positions[idx]
            actions = actions[idx]
            rewards = rewards[idx]


        observation_mean = positions.mean(axis=0)
        observation_std = positions.std(axis=0)
        observation_std = np.where(np.abs(observation_std) < 1e-6, 1.0, observation_std)

        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        action_std = np.where(np.abs(action_std) < 1e-6, 1.0, action_std)

        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        if abs(reward_std) < 1e-6: reward_std = 1.0

        with open_dict(self.cfg):
            self.cfg.observation_shape = int(positions.shape[-1])
            self.cfg.observation_mean = observation_mean.tolist()
            self.cfg.observation_std = observation_std.tolist()
            self.cfg.action_mean = action_mean.tolist()
            self.cfg.action_std = action_std.tolist()
            self.cfg.reward_mean = reward_mean
            self.cfg.reward_std = reward_std
            self.cfg.grid_shape = int(grids.shape[-1])
            self.cfg._runtime_stats_initialized = True


    # def _update_cfg_stats_from_dataset(self):
    #     observations = self.dataset["observations"]
    #     actions = self.dataset["actions"]
    #     rewards = self.dataset["rewards"]

    #     # if observations.size == 0:
    #     #     return

    #     max_stats_samples =  200000
    #     n = observations.shape[0]
    #     if n > max_stats_samples:
    #         idx = np.random.default_rng(0).choice(
    #             n, size=max_stats_samples, replace=False
    #         )
    #         observations = observations[idx]
    #         actions = actions[idx]
    #         rewards = rewards[idx]

    #     observation_mean = observations.mean(axis=0)
    #     observation_std = observations.std(axis=0)
    #     observation_std = np.where(np.abs(observation_std) < 1e-6, 1.0, observation_std)

    #     action_mean = actions.mean(axis=0)
    #     action_std = actions.std(axis=0)
    #     action_std = np.where(np.abs(action_std) < 1e-6, 1.0, action_std)

    #     reward_mean = float(rewards.mean())
    #     reward_std = float(rewards.std())
    #     # if abs(reward_std) < 1e-6:
    #     #     reward_std = 1.0

    #     with open_dict(self.cfg):
    #         self.cfg.observation_shape = [int(observations.shape[-1])]
    #         self.cfg.observation_mean = observation_mean.tolist()
    #         self.cfg.observation_std = observation_std.tolist()
    #         self.cfg.action_mean = action_mean.tolist()
    #         self.cfg.action_std = action_std.tolist()
    #         self.cfg.reward_mean = reward_mean
    #         self.cfg.reward_std = reward_std
    #         self.cfg._runtime_stats_initialized = True
