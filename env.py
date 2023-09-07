"""
A trading environment compatible with Gym API.

@author: Riley Smith
Created: 08/31/2023
"""

from collections import defaultdict
import random
import string
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym import Env
from gym.spaces import Box, Discrete

# Use some stocks as only validation data
VALIDATION_STOCKS = ['AAPL', 'BA', 'IBM']

class StockTradingEnv(Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store stock history for reference
        self.stock_history = self._preprocess(self._sample_history())

        # Only two actions will be accepted: buy/hold or sell/sit
        self.action_space = Discrete(2)

        # Observation space is the actual stock history. 30 days with two values
        # for each day (adjusted close and volume). All values will be normalized
        # so that the maximum possible observation is 1 and the minimum is 0
        self.observation_shape = (30, 2)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                     high=np.ones(self.observation_shape),
                                     dtype=np.float32)
        
        # Set initial state as first 30 days worth of data
        self.state = self.stock_history[:30]
        self.day = 0

        # Set canvas for rendering
        self.canvas = np.ones((28*num_guesses, 28*5, 3), dtype=np.uint8)

        # Give intial dollar value (enough to buy 10 shares)
        self.money = 10
        self.shares_owned = 0

        # Track portfolio value (for giving rewards)
        self.portfolio_value = 10

    def _sample_history(self, val=False):
        """
        Choose a random stock from available data and random starting point in 
        time. Use this as the history for this environment.

        Returns
        -------
        stock_history : ndarray
            A Numpy array of 60 days worth of stock history. The first 30 days
            will be what is initially ingested by the network, and it will act
            as a trader of the stock over a 30 day period. Shape of data should
            be (60, 2) -- 60 days by two fields (closing price and volume).
        """
        # If requesting validation, get one of the validation-only stocks
        if val:
            idx = np.random.randint(0, 3)
            ticker = VALIDATION_STOCKS[idx]
            csvfile = str(Path('data', f'{ticker}.csv'))
        else:
            # List available stock CSV files
            available_files = sorted(list(Path('data').glob('*.csv')))
            random.shuffle(available_files)
            csvfile = str(available_files[0])
        # Load the data
        data = pd.read_csv(csvfile, usecols=['Adj Close', 'Volumne'])
        # Choose random starting point
        first_day = np.random.randint(0, data.shape[0] - 60)  
        return data.iloc[first_day: first_day + 60].to_numpy()

    def _preprocess(self, history):
        """
        Apply simple preprocessing so that each sequence is like an index.
        That is, the 30th day is the reference ($1) and every other day is
        given relative to this price.
        """
        # Get reference data
        ref_data = history[29][np.newaxis, :]
        return history / ref_data

        # # Get min/max only from first 30 day period
        # initial_data = history[:30]
        # self.h_min = initial_data.min(axis=0)
        # self.h_max = initial_data.max(axis=0)
        # self.h_ptp = self.h_max - self.h_min
        # return (history - self.h_min[np.newaxis,:]) / self.h_ptp[np.newaxis, :]

    def draw_on_canvas(self, guess):
        """Update the canvas to show the latest guess"""
        # Get state for each letter and calculate reward (2 points for green, 1 for yellow, 0 for gray)
        tile_states = []
        new_state = []
        letters_guessed = defaultdict(int)

        for guessed_letter, true_letter in zip(guess, self.answer):
            if guessed_letter == true_letter:
                tile_state = 2
                letters_guessed[guessed_letter] += 1
                letter_position = string.ascii_lowercase.index(guessed_letter) + 1
                local_state = 52 + letter_position
            else:
                tile_state = -1
                local_state = -1
            new_state.append(local_state)
            tile_states.append(tile_state)

        for i, (guessed_letter, tile_state) in enumerate(zip(guess, tile_states)):
            if tile_state != -1:
                continue
            letters_guessed[guessed_letter] += 1
            num_occurrences = self.answer.count(guessed_letter)
            if letters_guessed[guessed_letter] <= num_occurrences:
                tile_state = 1
            else:
                tile_state = 0
            # Now account for which letter of the alphabet it is
            letter_position = string.ascii_lowercase.index(guessed_letter) + 1
            # Compute state. First 26 are gray, next 26 yellow, last 26 green
            local_state = tile_state * 26 + letter_position
            new_state[i] = local_state
            # Also store the tile_state
            tile_states[i] = tile_state

        rendered_tiles = [plot_tile(letter, color=state) for letter, state in zip(guess, tile_states)]
        new_row = np.concatenate(rendered_tiles, axis=1)
        self.canvas[(self.guesses - 1) * 28: self.guesses * 28, :, :] = new_row

    def step(self, action, display=False):
        """
        Step function for custom trading environment.

        Each day, reward is change in portfolio value from previous day.

        
        Take the given action (integer from 0 to 2309) and retrieve the word
        corresponding to that guess. Compute the new state based on that word
        and the answer.

        Parameters
        ----------
        action : int
            The integer for the index of the word guessed.
        """
        # Use action to update holdings
        latest_price = self.state[-1,0]
        if action == 1:
            # This means buy if not owned, or hold if owned
            while self.money > latest_price:
                self.shares_owned += 1
                self.money -= latest_price
        else:
            # This means sell if owned, or hold if not
            while self.shares_owned > 0:
                self.shares_owned -= 1
                self.money += latest_price
        
        # Now compute new portfolio value
        new_value = (self.shares_owned * latest_price) + self.money
        reward = new_value - self.portfolio_value
        self.portfolio_value = new_value

        # And update state
        self.day += 1
        self.state = self.stock_history[self.day: self.day + 30]

        # Check to see if done
        terminated = self.day > 30

        # Environment will return a hard-coded "False" to be consistent with
        # changes to OpenAI Gym API, which now has separate indicator for
        # truncated or terminated. But, I make no distinction, therefore in my
        # case, truncated is always "False"
        return self.state, reward, terminated, False, {}

    def render(self):
        plt.imshow(self.canvas, vmin=0, vmax=255)
        plt.show()
        time.sleep(2)

    def reset(self, val=False):
        # Reset state and holdings
        self.stock_history = self._preprocess(self._sample_history(val))
        self.state = self.stock_history[:30]
        self.day = 0
        self.money = 10
        self.shares_owned = 0
        self.portfolio_value = 10
        return self.state, {}
#
# env = WordleEnv()
# env.state
# env.action_space.n
# dir(env.action_space)
#
# test = np.array([[1, 2], [3, 4]])
# np.array([1, 2]) in test
#
# env.step(env.action_space.sample())
#
#
# env.render()
#
# env.state[np.where(env.state > 0)].shape
#
# sub = env.state[np.where(env.state[:,0] > 0)]
# (sub % 26)[:,0]
#
# from importlib import reload
# reload(agents)
#
# def test_env():
#     env = WordleEnv()
#     state = env.state
#     print('Answer is: ', env.answer)
#
#     done = False
#     while not done:
#         action = agents.fixed_score_agent(state, WORDS)
#         # action = env.action_space.sample()
#         state, _, done, info = env.step(action)
#         env.render()
#         print('Guess: ', info['guess'])
#
# test_env()
