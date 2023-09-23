# DNN-stock-trader
A comparison of LSTM vs Deep Q-learning for a simple stock trader, and an
explanation of why data used is insufficient for accurate results.

## LSTM Networks
LSTM (long short-term memory) networks are a popular framework of recurrent
neural networks. They ingest sequences of data and predict some outcome variable
based on those sequential inputs. In the context of stock trading, the input
sequence could be a sequence of historical prices and/or other statistics about
the stock, like trading volume. The target variable could then be some
forward-looking measure of stock performance, such as the change in price over
the course of the next day, week, or month.

LSTM networks have been shown to be effective for many kinds of sequential
modeling problems, but are not without issues. Like vanilla recurrent neural
networks (RNNs), LSTMs can suffer from vanishing or exploding gradients from
long sequences, though this problem is greatly mitigated with LSTM architecture.
More recent work in sequence modeling has favored attention mechanisms, such as
the original Transformer network and derivatives from it, but LSTMs are still
useful for a variety of tasks.

## Deep Q-Learning
Deep Q-Learning is entirely different to traditional deep learning with LSTMs.
It is a reinforcement learning technique that consists of two networks, often
termed a "Q-network" and a "target network". Essentially the Q-network takes in
states and predicts the Q-values of each action for the given state. The actual
reward from the reinforcement learning environment and the "true" Q-values from
the target network are used to correct (train) the Q-network. In reality, the
target network does not start with true Q-values, but learns them over time by
periodically copying the weights from the Q-network into the target network.

For the problem of stock trading, this works as follows. The input data is still
a sequence of historical stock prices. Based on this data, the Q-network can
decide to buy the stock (or hold if already owned) or sell the stock (or hold
if not owned). A reward is then calculated based on the change of portfolio
value for the Deep Q-Learning agent. 

While this problem could get very complicated with the introduction of different
choices of stocks and financial assets, for simplicity this analysis focuses on
allowing the agent to trade only one stock at a time.

## Data
Data for this analysis is scraped from Yahoo Finance with a simple scraper that
downloads CSV files of historical stock data and saves them. Only price and
volume data is available, which severely limits this analysis (see "Results"
section below). Training data is scraped for stocks in the S&P 500 only and
three stocks are chosen to be used as holdout data (Apple, Boeing, and IBM).
Each sequence presented to the LSTM or RL agent is limited to 30 days of
historical data and the network is always trying to maximize next-day profits.

To normalize inputs and make the network task easier, each sequence is normalized
so that the price on the last day of input data (day 30) is 1. All other prices
are indexed relative to this value. This way, if a label (price on day 31) is,
say, 1.05, this can be directly interpreted as a 5% increase in the stock price.
This normalization is applied for training both LSTM and RL agent.

# Results

### LSTM
Unsurprisingly the LSTM does not perform well in this experiment. With only
historical prices to learn from, day-to-day stock fluctuations are too random
for any meaningful learning to take place. In fact, if this model did perform
well, it would be very easy for someone to 

