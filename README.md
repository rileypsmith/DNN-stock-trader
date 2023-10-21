# DNN-stock-trader
A comparison of LSTM vs Deep Q-learning for a simple stock trader, and an
explanation of why data used is insufficient for accurate results. The point of
the analysis that follows is to explore how vanilla LSTM networks and LSTM-based
reinforcement learning may be used for intelligent sequence modeling and to
explore the different ways these methods lend to setting up and solving a
simple problem like stock trading. Ultimately this analysis will not produce a
protifable stock trader and I discourage anyone from using these results as the
basis for actual stock trading.

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

In this analysis, LSTMs are used in two different ways, which are discussed below.

### Auto-regressive LSTM 

The first approach is to take a
sequence of historical price data and use an LSTM to directly predict the next
day's asset price. In this case, the output of the model is a float and the model is trained using mean squared error as a loss function. Once trained, this model can then be used auto-regressively to predict
prices some time in the future, though as we will see, historical prices are
not sufficient fo build an accurate model in this fashion.

### Classification LSTM

A second approach is to discretize stock returns over some fixed time horizon
and turn the sequence modeling task into a classificaiton task: in this case,
the LSTM ingests historical prices and tries to predict which "bin" the stock
returns will fall into over the following 'n' days. For this analysis, five bins are used, corresponding to stock returns in $(-\inf, -0.03],(-0.03,-0.01],(-0.01,0.01],(0.01,0.03],(0.03,\inf)$. The time horizon over which returns are predicted can be anything, but in this analysis we will attempt to predict stock prices 5 days in the future. 

For training, a different loss function is required than the auto-regressive
approach since this approach is more of a classification task. In general,
cross-entropy loss is useful for classification tasks, but standard sparse
categorical crossentropy assumes there is no relation between neighboring
classes (e.g. dog vs cat--the fact that they may be classes 0 and 1 is irrelevant).
In our case, however, neighboring classes are more similar than distance classes.
If a stock actually returns 4% and is predicted to return 2%, this is much better
than if it is predicted to return -2%. To capture this, a custom loss function is
used that first creates a target vector for each class that is not a one-hot
vector: instead, it contains the highest value for the correct class but decays
exponentially as class index moves away from this point. Target vectors are
normalized to they sum to 1, i.e. so they encode a probability distribution over
classes. The final output of the network (a softmax layer) is then compared to
this target vector via cross-entropy and the loss is as follows:

$L=-1 *  {\huge \sum}\limits_{i=0}^n t_i ln(p_i)$, where $t$ is the target vector, $p$ is
the predicted probability vector, and $n$ is the number of classes.

The classification approach fairs somewhat
better than the auto-regressive approach, but is still a terrible basis for
trading.

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
allowing the agent to trade only one stock at a time. Additionally, the agent is
limited to two actions at each step: buy or sell. If the agent chooses the buy
action and the asset is already owned, this is interpreted as a hold and no
action is taken. Similarly, if sell is chosen but the asset is not owned, the
agent takes no ation. Note that this assumes the agent is investing all of its
available capital at one time; it cannot choose to invest only partial capital
(this would significantly complicate the problem, though may work alongside the
choice of multiple assets, and could be a possible extension of this analysis).

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

## LSTM

Unsurprisingly the LSTM does not perform well in this experiment. With only
historical prices to learn from, day-to-day stock fluctuations are too random
for any meaningful learning to take place. In fact, if this model did perform
well, it would be very easy for anyone to recreate these results, and the
LSTM predictions would be accounted for by future-looking trades in the
market, but this is a digression.

### Auto-regressive implementation

Let's take a look at the performance of the auto-regressive model. This model is
trained to ingest 30 days worth of historical stock prices and output a float
prediction for the next day's closing price. It can then be evaluated in an
auto-regressive fashion: allow it to predict the next day's price, then feed
this predicted price back in as the last datapoint to allow it to predict the
price two days out, and so on. When evaluating performance, we can "supervise"
the network by correcting it with the actual closing price every 'n' days to see
how many days in the future it can make accurate predictions before breaking down.

Below is a plot of what happens when a trained network makes predictions one day
at a time for one of the holdout stocks, Apple.

![AAPL true vs. pred](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/auto-regressive_LSTM/EPOCH005_AAPL_01.png)

This looks pretty good, right? Well, not so much. In this case, the network is
being corrected every day. So it is only ever asked to make predictions one day
in the future. It may be hard to tell from this plot, but basically the network
is learning to always predict a slight increase in the stock price. This makes
sense in the context of historical stock pricess--they have gone up considerably
over the time frame covered by the dataset. But it does little to help us
predict day-to-day fluctuations.

We can see this better by looking at results that are corrected only every 10
days. Here is that result for another holdout stock, Boeing:

![BA true vs. pred](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/auto-regressive_LSTM/EPOCH005_BA_10.png)

Now we can see that when the network is allowed to make its own predictions into
the future, it basically just interpolates an estimate that the stock will
appreciate over time. Again, not bad on average over a very long time period, 
but in this example we can see how it would hurt us if we used this prediction
to inform trading decisions regarding Boeing.

In some cases though, this interpolation is surprisingly good. Here is an exmaple
from a third holdout stock (IBM) that is allowed to predict 30 days into the
future auto-regressively:

![BA true vs. pred](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/auto-regressive_LSTM/EPOCH005_IBM_30.png)

We see the same trend of interpolated expectations for appreciation, but in this
case it is surprisingly not bad.

The takeaway of all this is that with only historical prices to learn from, the
LSTM is only able to learn that over time stocks increase in value, which is true
within the context of the dataset. But it does not have enough information to
learn more high-frequency fluctuations in the stock price--at least not with only
30 days of historical data to look at.

### Classification approach

Now let's take a look at the performance of the classification model. Results
in the plot below show performance over the course of training on a holdout set
looking 5 days into the future. A prediction of class 4 represents highest
estimated returns (>3%) while a prediction of class 0 represents lowest possible
returns (loss > 3%). 

![BA true vs. pred](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/classification_LSTM/5day_classification_lstm_results.png)

There are two main things to note in this plot. First,
we can see that the classes are roughly sorted correctly by the end of training.
That is, on average stocks predicted in class 4 outperform those in class 0. 
However, the error bars show one standard deviation around the mean returns for
each class, and this is not pretty. Although the model is on average learning
to more or less sort stocks correctly according to their forward looking returns,
there is massive uncertainty. If this model were used to actually try and trade
these stocks, returns would swing wildly from large profits to large losses and
the risk would be very high.

Again we are faced with the same conclusion: historical prices alone are not
sufficient data to build an accurate model for future asset price forecasting.

## Deep Q-learning
So LSTM networks are not well equipped for this task, at least not gien the
quickly scraped data. How did the RL agent fare?

If you guessed not well, you are correct. Unsurprisingly, the reinforcement
learning agent is just as handicapped by data limitations as the vanilla LSTMs.
After training for 128,000 decision steps, the RL agent has not converged to any
consistent level of trading performance. 

Let's take a look at just what it learns. Below is a plot of the portfolio value
of the RL agent over a 30-day trading period from a starting value of $10.
This is the average result of 100 trials (i.e. the agent is given 100 random
sets of data from the holdout stocks and allowed to trade for 30 days, and the
results are averaged).

Here is that result from epoch 1 of training, after 1,280 decision steps:

![RL epoch 1](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/deep_Q_learning/EPOCH000.png)

This doesn't appear to be moving at all. So what's happening here? Well, early
in training, the network is mostly randomly making decisions due to a high rate
of exploration forced on the agent through an epsilon-greedy policy. A parameter
"epsilon" controls how often the agent takes the decision it thinks is best vs
how often it takes a random action. This value starts high in training. Over the
course of training, it is steadily decreased (but never to 0, so some exploration
is always taking place). As a result, early outcomes are totally unpredictable
and the agent seems to learn to take no action in the early stages of training.

Here is the result after 10 epochs of training (12,800 decision steps):

![RL epoch 10](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/deep_Q_learning/EPOCH010.png)

Now we see that the agent is choosing to sometimes trade a stock, though 
interestingly it never makes a trade in the first 10 days of the 30-day trading
period. Uncertainty is high, and although on average the agent gains 1% over the
30-day trading window, the 95% confidence interval on its performance spans from
-4% to +4%.

Did it improve by the end of training? Not really. Here is the final result,
after training for 100 epochs (128,000 decision steps):

![RL epoch 100](https://github.com/rileypsmith/DNN-stock-trader/blob/main/plots/deep_Q_learning/EPOCH099.png)

The agent still never likes to trade in the first few days (it is not immediately
clear why). This time, the agent actually loses money on average, ending down about
0.1% on average, a small amount but still not a good result. The only "improvement"
over the result from 10 epochs of training is that the error bars are slightly
narrower, but it can't be concluded that this is as a result of training for
longer and not just a result of the configuration of weights that existed at the
instance that training was stopped.

The bottom line of the reinforcement learning approach is the same as the main
takeaway from the LSTM approach: better data is needed to achieve practical
results.

# Discussion and Future Analysis
This analysis has been intended to explore two different ways to implement
sequence-modeling with neural networks: standard LSTMs and LSTM-based reinforcement
learning agents with deep-Q learning. The code contained in this repository
can be informative as a simple starting point for setting up these kinds of
networks, but the results are not practical. Since the focus of this analysis
was not to build an actually profitable trading bot, more emphasis was placed
on the training setup of the neural networks and less was placed on data
acquisition. In fact, the problem of developing a neural network capable of
profiting in the stock market is widely studied and elusive; any attempt to
actually accomplish this would certainly require a much richer dataset than
that which can be readily scraped from Yahoo Finance. 

In short, both approaches used in this analysis suffer from data limitations.
This analysis has not reached the predictive limitations of LSTM networks or
the design limitations of reinforcement learning problems, but better results
can likely only be achieved with better data.




