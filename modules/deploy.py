import json
import torch
import numpy as np
import gym

from modules.td3 import TD3


class Deploy:
    def __init__(self, parameters):
        self.env = gym.make(parameters.get('env_name'))
        self.env.seed(parameters.get('seed'))
        torch.manual_seed(parameters.get('seed'))
        np.random.seed(parameters.get('seed'))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.file_name = "%s_%s" % ("TD3", parameters.get('env_name'))

        print(f"state dim: {self.state_dim} \n  action_dim: {self.action_dim} \n max_action: {self.max_action}")

        self.policy = TD3(self.state_dim, self.action_dim, self.max_action)
        self.policy.load(self.file_name, f"pytorch_models/{self.file_name}")

    def run_policy(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.policy.select_action(np.array(obs))
                obs, reward, done, _ = self.env.step(action)
                self.env.render()
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

