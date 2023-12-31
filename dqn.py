"""
A stock trader based on Deep Q-learning.

@author: Riley Smith
Created: 08/31/2023
"""
import csv
from pathlib import Path
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Sequential

from env import StockTradingEnv
import utils

warnings.simplefilter('ignore')

class Agent():
    """A class for containing q-net and t-net, plus all agent functions."""
    def __init__(self, use_volume=False):
        # Setup networks
        self.q_net = self._build_net(use_volume)
        self.t_net = self._build_net(use_volume)
        self.update()

        self.epsilon = 1
        self.gamma = 0.9
    
    def _build_net(self, use_volume):
        """Simple model which wil predict next day's returns"""
        input_shape = (30,2) if use_volume else (30,1)
        model = Sequential()
        model.add(layers.LSTM(128, return_sequences=True, 
                              kernel_initializer='he_normal', input_shape=input_shape))
        model.add(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'))
        model.add(layers.LSTM(128, return_sequences=False, kernel_initializer='he_normal'))
        model.add(layers.Dense(2, activation='linear'))
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
            run_eagerly=True
        )
        return model
    
    def summary(self):
        self.q_net.summary()

    def policy(self, state, train=True):
        """
        Given a state, decide on a course of action.
        
        Parameters
        ----------
        state : ndarray
            Ndarray of shape (30,2).
        """
        # Prepare for network
        net_input = tf.convert_to_tensor(state[np.newaxis,:,:])
        # Put it through the Q-Net
        q_est = self.q_net(net_input).numpy().ravel()
        # Choose optimal policy, or random policy according to epsilon
        if np.random.random() < self.epsilon and train:
            action = np.random.randint(0, 2)
        else:
            action = q_est.argmax()
        return action
    
    def train(self, batch):
        """
        Batch comes in as:

        state, action, reward, next state, done
        """
        # Setup network input
        states = tf.convert_to_tensor(np.stack([item[0] for item in batch], 
                                               axis=0))
        
        # Form 'truth' from next states and actual rewards
        next_states = tf.convert_to_tensor(np.stack([item[3] for item in batch], 
                                                    axis=0))
        truth = self.t_net(next_states).numpy()

        for i, (_, action, reward, _, done) in enumerate(batch):
            if done:
                truth[i, action] = reward
            else:
                truth[i, action] = reward + (self.gamma * truth[i, action])
        
        # Run SGD on the batch you just formed
        self.q_net.fit(states, truth, epochs=1)

    def run_episode(self, env, return_values=False):
        # Start by resetting the environment
        state, _ = env.reset(val=True)
        done = False
        total_reward = 0
        value_over_time = [env.portfolio_value]
        while not done:
            action = self.policy(state, train=False)
            state, reward, done, _, info = env.step(action)
            total_reward += reward
            value_over_time.append(env.portfolio_value)
        if return_values:
            return total_reward, value_over_time
        return total_reward

    def evaluate(self, env, saveas=None, num_episodes=10):
        """Run reward-based evaluation and make a plot"""
        values_over_time = []
        total_reward = 0
        for _ in tqdm(range(num_episodes)):
            reward, value_over_time = self.run_episode(env, return_values=True)
            total_reward += reward
            values_over_time.append(value_over_time)
        # Compute average reward
        av_reward = total_reward / num_episodes
        # Compute average portfolio value over time
        values_over_time = np.array(values_over_time)
        av_vals = values_over_time.mean(axis=0)
        confidence_interval = values_over_time.std(axis=0) * 1.96
        # Plot it
        fig, ax = plt.subplots()
        x = np.arange(av_vals.size)
        ax.plot(x, av_vals, color='black', linestyle='--', label='Average portfolio value')
        ax.fill_between(x, av_vals - confidence_interval, av_vals + confidence_interval,
                        color='orange', alpha=0.2, label='95% confidence interval')
        ax.legend()
        if saveas is not None:
            plt.savefig(saveas)
            plt.close()
        else:
            plt.show()
    
    def update(self):
        """Copy the weights of the Q network over to the target network"""
        self.t_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        """Every episode, update epsilon"""
        if self.epsilon > 0.05:
            self.epsilon = np.max([self.epsilon * 0.9, 0.05])

class ReplayBuffer():
    """A class to handle all things related to the replay buffer"""
    def __init__(self, buffer_size=1280):

        self.buffer = []
        self.buffer_size = buffer_size

    def store_experience(self, env, agent, state):
        # Implement agent's policy to get an action for this state
        action = agent.policy(state)
        # Get reward and next state from environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        self.buffer.append((state, action, reward, next_state, done))
        return next_state, done

    def fill_buffer(self, env, agent):
        progbar = tqdm(total=self.buffer_size)
        while len(self.buffer) < self.buffer_size:
            # Start by resetting environment
            state, _ = env.reset()
            done = False
            while not done:
                state, done = self.store_experience(env, agent, state)
                progbar.update(1)
                if len(self.buffer) >= self.buffer_size:
                    break

    def sample_buffer(self, batch_size=128):
        """Return a random sample from the buffer, without replacement"""
        for batch_num in range(0, self.buffer_size, batch_size):
            yield random.sample(self.buffer, batch_size)

    def update(self):
        self.replay_buffer = []

class CustomCSVWriter():
    """Super simple callback for writing to CSV file"""
    def __init__(self, csv_file):
        # Write headers to file
        self.csv_file = csv_file
        headers = ['Epoch', 'av_reward']
        with open(csv_file, 'w+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    def update(self, epoch, val_steps):
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, val_steps])

def train(outdir, num_epochs=100):
    # Build an agent
    agent = Agent()
    # Setup replay buffer
    replay_buffer = ReplayBuffer(buffer_size=2560)
    # Build environment
    env = StockTradingEnv()
    # Setup outputs
    outdir = utils.setup_output_dir(outdir)
    log = CustomCSVWriter(str(Path(outdir, 'DQN_log.csv')))
    # Make a directory for evaluation figures
    figdir = Path(outdir, 'DQN_eval')
    figdir.mkdir()
    # Run a bunch of epochs of training
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch:03}')
        replay_buffer.fill_buffer(env, agent)
        for batch in replay_buffer.sample_buffer(batch_size=128):
            agent.train(batch)
        # Run evaluation at the end of each epoch
        figpath = str(Path(figdir, f'EPOCH{epoch:03}.png'))
        eval_result = agent.evaluate(env, saveas=figpath)
        log.update(epoch, eval_result)
        print(f'Epoch {epoch+1}/{num_epochs}: {eval_result}')

        # Update agent and buffer
        replay_buffer.update()
        agent.update_epsilon()
        agent.update()

if __name__ == '__main__':
    outdir = 'DQN_output'
    train(outdir)

    


        