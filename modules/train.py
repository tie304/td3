import os
import json
import time
import gym
from utils.utils import pre_process_states
import torch
import numpy as np
#import pybullet_envs
from gym import wrappers

from utils.utils import mkdir
from modules.td3 import TD3
from modules.replay_buffer import ReplayBuffer


class Train:
    def __init__(self, parameters):
        self.parameters = parameters
        self.env = gym.make(parameters.get('env_name'))
        self.replay_buffer = ReplayBuffer()
        self.env.seed(parameters.get('seed'))
        torch.manual_seed(parameters.get('seed'))
        np.random.seed(parameters.get('seed'))
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.policy = TD3(self.state_dim, self.action_dim, self.max_action, parameters.get('train_with_conv'))
        if parameters.get('load_train_weights'):
            self.policy.load(parameters.get('load_train_weights'), f"pytorch_models/{parameters.get('load_train_weights')}")
        self.evaluations = []
        self.max_episode_steps = self.env._max_episode_steps

        print(f"state dim: {self.state_dim} \n  action_dim: {self.action_dim} \n max_action: {self.max_action}")

        self.file_name = "%s_%s" % ("TD3", parameters.get('env_name'))

        if not os.path.exists("./results"):
            os.makedirs("./results")
        if parameters.get('save_models') and not os.path.exists(f"./pytorch_models/{self.file_name}"):
            os.makedirs(f"./pytorch_models/{self.file_name}")

    def evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.policy.select_action(np.array(obs))
                obs, reward, done, _ = self.env.step(action)
                if self.parameters.get('eval_render'):
                    self.env.render()
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

    def train(self):
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        episode_times = []
        training_times = []
        done = True
        t0 = time.time()

        # We start the main loop over n timesteps
        while total_timesteps < self.parameters.get('max_timesteps'):
            # If the episode is done
            episode_start_time = time.time()
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    train_time = self.policy.train(self.replay_buffer, episode_timesteps, self.parameters.get('batch_size'), self.parameters.get('discount'), self.parameters.get('tau'),
                                 self.parameters.get('policy_noise'), self.parameters.get('noise_clip'),
                                 self.parameters.get('policy_freq'))

                    training_times.append(train_time)

                    avg_training_times = sum(training_times) / len(training_times)
                    avg_episode_times = sum(episode_times) / len(episode_times)
                    total_time = avg_episode_times + avg_training_times
                    max_timesteps = self.parameters.get('max_timesteps')

                    est_end = ((max_timesteps - total_timesteps) / episode_timesteps) * total_time


                    print(f"Estimated completion time:  {est_end / 60} minutes, {(est_end / 60 )/ 60} hours, days: {(est_end / 60) / 60 / 24}")

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= self.parameters.get('eval_freq'):
                    timesteps_since_eval %= self.parameters.get('eval_freq')
                    self.evaluations.append(self.evaluate_policy())
                    self.policy.save(self.file_name, directory=f"./pytorch_models/{self.file_name}")
                    np.save("./results/%s" % (self.file_name), self.evaluations)

                # When the training step is done, we reset the state of the environment
                obs = self.env.reset()
                if self.parameters.get('train_with_conv'):
                    obs = pre_process_states(obs)

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < self.parameters.get('start_timesteps'):
                action = self.env.action_space.sample()
            else:  # After 10000 timesteps, we switch to the model
                action = self.policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.parameters.get('expl_noise') != 0:
                    action = (action + np.random.normal(0, self.parameters.get('expl_noise'), size=self.env.action_space.shape[0])).clip(
                        self.env.action_space.low, self.env.action_space.high)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = self.env.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == self.env._max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            if self.parameters.get('train_with_conv'):
                new_obs = pre_process_states(new_obs)

            assert new_obs.shape == obs.shape, f"States dont match in size {new_obs.shape}, {obs.shape}"

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            self.replay_buffer.add((obs, new_obs, action, reward, done_bool))

            if self.parameters.get('training_render'):
                self.env.render()

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            if done:
                episode_times.append(time.time() - episode_start_time)

        # We add the last policy evaluation to our list of evaluations and we save our model
        self.evaluations.append(self.evaluate_policy())
        if self.parameters.get('save_models'): self.policy.save("%s" % (self.file_name), directory=f"./pytorch_models/{self.file_name}")
        np.save("./results/%s" % (self.file_name), self.evaluations)