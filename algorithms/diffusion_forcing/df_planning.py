from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
)


class DiffusionForcingPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        observation_shape_cfg = cfg.get("observation_shape", None)
        if observation_shape_cfg is not None:
            observation_shape = tuple(int(x) for x in observation_shape_cfg)
            self.observation_dim = int(np.prod(observation_shape))
        else:
            fallback_mean = cfg.get("observation_mean", None)
            if fallback_mean is None:
                raise ValueError(
                    "Missing observation shape information: expected cfg.observation_shape "
                    "or cfg.observation_mean."
                )
            self.observation_dim = int(np.array(fallback_mean).reshape(-1).shape[0])

        raw_observation_mean = np.array(cfg.observation_mean).reshape(-1)
        raw_observation_std = np.array(cfg.observation_std).reshape(-1)
        if raw_observation_mean.size == self.observation_dim:
            self.observation_mean = raw_observation_mean
        elif raw_observation_mean.size == 1:
            self.observation_mean = np.repeat(
                raw_observation_mean, self.observation_dim
            )
        else:
            self.observation_mean = np.zeros(self.observation_dim, dtype=np.float32)

        if raw_observation_std.size == self.observation_dim:
            self.observation_std = raw_observation_std
        elif raw_observation_std.size == 1:
            self.observation_std = np.repeat(raw_observation_std, self.observation_dim)
        else:
            self.observation_std = np.ones(self.observation_dim, dtype=np.float32)

        self.observation_std = np.where(
            np.abs(self.observation_std) < 1e-6, 1.0, self.observation_std
        )
        self.action_mean = np.array(cfg.action_mean).reshape(-1)
        self.action_std = np.array(cfg.action_std).reshape(-1)
        self.action_dim = int(self.action_mean.shape[0])
        self.use_reward = cfg.use_reward
        self.unstacked_dim = (
            self.observation_dim + self.action_dim + int(self.use_reward)
        )
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.open_loop_horizon = cfg.open_loop_horizon
        self.padding_mode = cfg.padding_mode
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0
        self.plot_env_id = (
            self.env_id
            if any(k in self.env_id for k in ("umaze", "medium", "large"))
            else "array_dataset"
        )

    def _build_model(self):
        # Added flatten to support non 1D inputs
        mean = list(self.observation_mean.flatten()) + list(self.action_mean)
        std = list(self.observation_std.flatten()) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.cfg.data_mean = np.array(mean).tolist()
        self.cfg.data_std = np.array(std).tolist()
        super()._build_model()

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError(
                "Number of frames - 1 must be divisible by frame stack size"
            )

        nonterminals = torch.cat(
            [
                torch.ones_like(nonterminals[:, : self.frame_stack]),
                nonterminals[:, :-1],
            ],
            dim=1,
        )
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(
            self.make_bundle(observations, actions, rewards)
        )  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = (
            0  # zero out actions and rewards after normalization
        )
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None

        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(
                2, n_tokens + 1, (batch_size,), device=self.device
            )
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[
                :, :n_tokens
            ].bool()
            random_terminal = repeat(
                random_terminal, "b t -> (t fs) b", fs=self.frame_stack
            )
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(
            xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks)
        )

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log(
                "training/loss", loss, on_step=True, on_epoch=False, sync_dist=True
            )

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = self._observations_to_xy(o).detach().cpu().numpy()[:-1, :8]
            # last observation is dummy, sample 8
            images = make_trajectory_images(
                self.plot_env_id, trajectory, trajectory.shape[1], None, None, False
            )
            for i, img in enumerate(images):
                self.log_image(
                    f"training_visualization/sample_{i}",
                    Image.fromarray(img),
                )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        if self.action_dim != 2:
            self.eval_planning(
                batch, conditions, horizon, namespace + str(horizon)
            )  # can run planning without environment installation
        self.interact(batch, conditions, namespace)

    def plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        conditions: Optional[Any] = None,
    ):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan

        batch_size = start.shape[0]

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = (
                pred.shape[0] - self.frame_stack
            )  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(
                    pred, target, reduction="none"
                )  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20]
                    * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [
                        1 for _ in range(horizon)
                    ]  # try to reach the goal at any horizon
                    + [0]
                    * (
                        h_padded - horizon
                    )  # don't guide padded entries due to horizon % frame_stack != 0
                )
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = torch.from_numpy(weight).float().to(self.device)

                dist_o, dist_a, _ = self.split_bundle(
                    dist
                )  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()

                # Legacy maze2d states use paired observation channels (e.g. x/y, vx/vy).
                # Flattened grid observations (e.g. 147-d) are not pair-structured.
                if (self.observation_dim % 2 == 0) and (
                    not self._is_grid_observation()
                ):
                    dist_o = reduce(
                        dist_o,
                        "t b (n c) -> t b n",
                        "sum",
                        n=self.observation_dim // 2,
                    ).sqrt()
                else:
                    dist_o = torch.sum(dist_o, -1, keepdim=True).sqrt()

                dist_o = torch.tanh(
                    dist_o / 2
                )  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = torch.cat([dist_o, dist_a], -1)
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]

                episode_return = -(dist * weight).mean() * 1000
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError(
                    "reward guidance not officially supported yet, although implemented"
                )
                rewards = pred[:, :, -1]
                weight = np.array(
                    [10] * self.frame_stack
                    + [0.997**j for j in range(h)]
                    + [0] * h_padded
                )
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        chunk = torch.randn(
            (plan_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )
        chunk = torch.clamp(
            chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
        )
        pad = torch.zeros(
            (pad_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        plan_hist = [plan.detach()[: self.n_tokens - pad_tokens]]
        stabilization = 0
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            to_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = repeat(from_noise_levels, "t -> t b", b=batch_size)
            to_noise_levels = repeat(to_noise_levels, "t -> t b", b=batch_size)
            plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
                plan,
                conditions,
                from_noise_levels,
                to_noise_levels,
                guidance_fn=guidance_fn,
            )[1 : self.n_tokens - pad_tokens]
            plan_hist.append(plan.detach()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(
            plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack
        )
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        return plan_hist

    def eval_planning(
        self, batch, conditions=None, horizon=None, namespace="validation"
    ):
        start_obs, goal_obs, start_xy, goal_xy = self._extract_start_goal_from_batch(
            batch
        )

        start_normalized = self.split_bundle(
            self._normalize_x(self.make_bundle(start_obs))
        )[0]
        goal_normalized = self.split_bundle(
            self._normalize_x(self.make_bundle(goal_obs))
        )[0]

        horizon = self.episode_len if horizon is None else horizon
        plan_hist = self.plan(start_normalized, goal_normalized, horizon, conditions)
        plan = self._unnormalize_x(plan_hist[-1])
        plan = plan[self.frame_stack - 1 :]

        # Visualization
        o, _, _ = self.split_bundle(plan)
        o_xy = self._observations_to_xy(o).detach().cpu().numpy()[:-1, :16]
        maze_grids = None
        if self._is_grid_observation():
            wall, _, _ = self._decode_grid(start_obs)
            maze_grids = self._walls_to_maze_grids(wall[: o_xy.shape[1]])
        images = make_trajectory_images(
            self.plot_env_id,
            o_xy,
            o_xy.shape[1],
            start_xy[:16].tolist(),
            goal_xy[:16].tolist(),
            self.plot_end_points,
            maze_grids=maze_grids,
        )
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_plan/sample_{i}",
                Image.fromarray(img),
            )

    def interact(self, batch, conditions=None, namespace="validation"):
        start_obs, goal_obs, start_xy, goal_xy = self._extract_start_goal_from_batch(
            batch
        )
        if self._is_grid_observation():
            self._interact_grid(
                start_obs, goal_obs, start_xy, goal_xy, conditions, namespace
            )
            return

        try:
            import d4rl
            import gym
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print(
                "d4rl import not successful, skipping environment interaction. Check d4rl installation."
            )
            return

        print("Interacting with environment... This may take a couple minutes.")

        use_diffused_action = False
        if self.action_dim != 2:
            # https://arxiv.org/abs/2205.09991
            print(
                "Detected reduced observation/action space, using Diffuser like controller."
            )
        else:
            print(
                "Detected full observation/action space, using MPC controller w/ diffused actions."
            )
            use_diffused_action = True

        batch_size = start_obs.shape[0]
        envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
        envs.seed(0)

        terminate = False
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        obs = envs.reset()

        obs = torch.from_numpy(obs).float().to(self.device)
        start = obs.detach()
        obs_normalized = (
            (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
        ).detach()

        goal = np.concatenate(envs.get_attr("goal_locations"))
        goal = torch.Tensor(goal).float().to(self.device)
        goal = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        all_plan_hist = (
            []
        )  # a list of plan histories, each history is a collection of m diffusion steps

        # run mpc with diffused actions
        while not terminate and steps < self.episode_len:
            plan_hist = self.plan(
                obs_normalized, goal_normalized, self.episode_len - steps, conditions
            )
            plan_hist = self._unnormalize_x(plan_hist)  # (m t b c)
            plan = plan_hist[-1]  # (t b c)

            all_plan_hist.append(plan_hist.cpu())

            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    plan_vel = (
                        plan[t, :, :2] - plan[t - 1, :, :2]
                        if t > 0
                        else plan[t, :, :2] - obs[:, :2]
                    )
                    action = 12.5 * (plan[t, :, :2] - obs[:, :2]) + 1.2 * (
                        plan_vel - obs[:, 2:]
                    )
                action = torch.clip(action, -1, 1).detach().cpu()
                obs, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [
                    torch.from_numpy(item).float() for item in [obs, reward, done]
                ]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                obs_normalized = (
                    (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
                ).detach()

                steps += 1

        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())

        # Visualization
        samples = min(16, batch_size)
        trajectory = torch.stack(trajectory)
        start = start[:, :2].cpu().numpy().tolist()
        goal = goal[:, :2].cpu().numpy().tolist()
        images = make_trajectory_images(
            self.plot_env_id, trajectory, samples, start, goal, self.plot_end_points
        )

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

    def _is_grid_observation(self):
        return self._grid_layout() is not None

    def _grid_layout(self):
        side = int(np.sqrt(self.observation_dim))
        if side * side == self.observation_dim:
            return 1, side

        if self.observation_dim % 3 == 0:
            side_sq = self.observation_dim // 3
            side = int(np.sqrt(side_sq))
            if side * side == side_sq:
                return 3, side

        return None

    def _grid_side(self):
        layout = self._grid_layout()
        if layout is None:
            raise ValueError("Not a supported grid observation layout")
        return layout[1]

    def _extract_start_goal_from_batch(self, batch):
        observations = batch[0][..., : self.observation_dim].float().to(self.device)
        start_obs = observations[:, 0]
        goal_obs = observations[:, -1]
        start_xy = self._observations_to_xy(start_obs).detach().cpu().numpy()
        goal_xy = self._goals_to_xy(goal_obs).detach().cpu().numpy()
        return start_obs, goal_obs, start_xy, goal_xy

    def _observations_to_xy(self, obs):
        if not self._is_grid_observation():
            return obs[..., :2]

        layout = self._grid_layout()
        if layout is None:
            raise ValueError("Not a supported grid observation layout")
        channels, side = layout
        if channels == 3:
            flat = obs.reshape(*obs.shape[:-1], 3, side, side)[..., 1, :, :].reshape(
                *obs.shape[:-1], -1
            )
        else:
            grid = obs.reshape(*obs.shape[:-1], side, side).reshape(*obs.shape[:-1], -1)
            score_agent = -torch.minimum((grid - 2.0) ** 2, (grid - 4.0) ** 2)
            flat = score_agent

        idx = flat.argmax(-1)
        x = (idx // side).float()
        y = (idx % side).float()
        return torch.stack([x, y], -1)

    def _goals_to_xy(self, obs):
        if not self._is_grid_observation():
            return obs[..., :2]

        layout = self._grid_layout()
        if layout is None:
            raise ValueError("Not a supported grid observation layout")
        channels, side = layout
        if channels == 3:
            flat = obs.reshape(*obs.shape[:-1], 3, side, side)[..., 2, :, :].reshape(
                *obs.shape[:-1], -1
            )
        else:
            grid = obs.reshape(*obs.shape[:-1], side, side).reshape(*obs.shape[:-1], -1)
            score_goal = -torch.minimum((grid - 3.0) ** 2, (grid - 4.0) ** 2)
            flat = score_goal

        idx = flat.argmax(-1)
        x = (idx // side).float()
        y = (idx % side).float()
        return torch.stack([x, y], -1)

    def _decode_grid(self, obs):
        layout = self._grid_layout()
        if layout is None:
            raise ValueError("Not a supported grid observation layout")
        channels, side = layout
        if channels == 3:
            grid = obs.reshape(obs.shape[0], 3, side, side)
            wall = grid[:, 0] >= 254.0
        else:
            grid = obs.reshape(obs.shape[0], side, side)
            wall = (grid > 0.5) & (grid < 1.5)

        pos = self._observations_to_xy(obs).long()
        goal = self._goals_to_xy(obs).long()
        return wall, pos, goal

    def _encode_grid(self, wall, pos, goal):
        batch_size, side, _ = wall.shape
        layout = self._grid_layout()
        if layout is None:
            raise ValueError("Not a supported grid observation layout")
        channels, _ = layout
        b = torch.arange(batch_size, device=wall.device)

        if channels == 3:
            obs = torch.zeros((batch_size, 3, side, side), device=wall.device)
            obs[:, 0] = wall.float() * 255.0
            obs[b, 1, pos[:, 0], pos[:, 1]] = 1.0
            obs[b, 2, goal[:, 0], goal[:, 1]] = 1.0
            return obs.flatten(1)

        obs = torch.zeros((batch_size, side, side), device=wall.device)
        obs[wall] = 1.0
        obs[b, goal[:, 0], goal[:, 1]] = 3.0
        same = (pos[:, 0] == goal[:, 0]) & (pos[:, 1] == goal[:, 1])
        obs[b, pos[:, 0], pos[:, 1]] = 2.0
        obs[b[same], pos[same, 0], pos[same, 1]] = 4.0
        return obs.flatten(1)

    def _walls_to_maze_grids(self, wall):
        wall_np = wall.detach().cpu().numpy()
        maze_grids = []
        for sample in wall_np:
            maze_grids.append(
                [
                    "".join(
                        "#" if sample[i, j] else "O" for j in range(sample.shape[1])
                    )
                    for i in range(sample.shape[0])
                ]
            )
        return maze_grids

    def _step_grid_positions(self, pos, action, wall):
        next_pos = pos.clone()
        next_pos[action == 0, 0] -= 1
        next_pos[action == 1, 0] += 1
        next_pos[action == 2, 1] -= 1
        next_pos[action == 3, 1] += 1
        next_pos[:, 0] = next_pos[:, 0].clamp(0, wall.shape[1] - 1)
        next_pos[:, 1] = next_pos[:, 1].clamp(0, wall.shape[2] - 1)
        blocked = wall[
            torch.arange(wall.shape[0], device=wall.device),
            next_pos[:, 0],
            next_pos[:, 1],
        ]
        next_pos[blocked] = pos[blocked]
        return next_pos

    def _interact_grid(
        self, start_obs, goal_obs, start_xy, goal_xy, conditions, namespace
    ):
        print("Interacting with array-grid dynamics (no gym dependency).")

        batch_size = start_obs.shape[0]
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]

        obs = start_obs.detach().clone()
        goal_obs = goal_obs.detach().clone()
        wall, pos, goal = self._decode_grid(obs)
        obs_normalized = ((obs - obs_mean[None]) / obs_std[None]).detach()
        goal_normalized = ((goal_obs - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []
        terminate = False
        while not terminate and steps < self.episode_len:
            plan_hist = self.plan(
                obs_normalized, goal_normalized, self.episode_len - steps, conditions
            )
            plan_hist = self._unnormalize_x(plan_hist)
            plan = plan_hist[-1]

            for t in range(self.open_loop_horizon):
                if t >= plan.shape[0]:
                    terminate = True
                    break
                _, action, _ = self.split_bundle(plan[t])
                action_discrete = torch.round(action[:, 0]).long().clamp(0, 3)
                pos = self._step_grid_positions(pos, action_discrete, wall)
                reward_t = (pos == goal).all(-1).float()

                reward_np = reward_t.detach().cpu().numpy()
                reached = np.logical_or(reached, reward_np >= 1.0)
                episode_reward += reward_np
                episode_reward_if_stay += np.where(~reached, reward_np, 1)
                first_reach += ~reached

                obs = self._encode_grid(wall, pos, goal)
                obs_normalized = ((obs - obs_mean[None]) / obs_std[None]).detach()
                trajectory.append(
                    self.make_bundle(obs, action, reward_t[:, None]).cpu()
                )

                steps += 1
                if reward_t.all() or steps >= self.episode_len:
                    terminate = True
                    break

        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())

        if not trajectory:
            return

        samples = min(16, batch_size)
        trajectory = torch.stack(trajectory)
        traj_xy = self._observations_to_xy(self.split_bundle(trajectory)[0]).numpy()
        images = make_trajectory_images(
            self.plot_env_id,
            traj_xy,
            samples,
            start_xy.tolist(),
            goal_xy.tolist(),
            self.plot_end_points,
            maze_grids=self._walls_to_maze_grids(wall[:samples]),
        )
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def _generate_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(
                0, self.timesteps // 4, (batch_size,), device=xs.device
            )

        return noise_levels
