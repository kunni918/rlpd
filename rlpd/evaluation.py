from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)

    mean_return = np.mean(env.return_queue)
    normalized_mean_return = (mean_return - env.ref_min_score) / (env.ref_max_score - env.ref_min_score)
    return {"return": normalized_mean_return * 100. , "length": np.mean(env.length_queue)}
