"""
A simple LSTM to learn from historical stock data and decide whether to
buy, sell, or hold for a stock daily over a period of time.

@author: Riley Smith
Created: 08/30/2023
"""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers, Sequential

warnings.simplefilter('ignore')

# Use some stocks as only validation data
VALIDATION_STOCKS = ['AAPL', 'BA', 'IBM']

class DirectionalAccuracy(tf.keras.metrics.Metric):
    """
    A custom metric to track how often the network is correct about the 
    direction of change of a stock price.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.correct_count = 0
        self.total_count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        comparison = tf.cast(tf.equal(tf.math.sign(y_true), tf.math.sign(y_pred)), tf.int32)
        self.correct_count += tf.math.reduce_sum(comparison)
        self.total_count += tf.size(y_true)

    def result(self):
        return self.correct_count / self.total_count
    
    def reset_state(self):
        self.correct_count = 0
        self.total_count = 0

def load_and_preprocess(csvfile):
    """Load and preprocess the data for one stock"""
    data = pd.read_csv(csvfile, usecols=['Adj Close', 'Volume'])
    # Break it up into sequences of 30 days worth of data
    sequences = []
    labels = []
    for i in range(0, data.shape[0] - 30, 30):
        sequence = data.iloc[i: i + 30].to_numpy()
        # Normalize with min/max scaling
        s_min = sequence.min(axis=0)
        s_max = sequence.max(axis=0)
        sequence = (sequence - s_min[np.newaxis,:]) / (s_max[np.newaxis,:] - s_min[np.newaxis,:])
        # Get label and make sure it is also normalized
        tgt_price = (data.loc[i + 30, 'Adj Close'] - s_min[0]) / (s_max[0] - s_min[0])
        label = tgt_price - sequence[-1, 0]
        # If any are NaN, don't keep it. This occurs when there is an error in
        # reporting the stock price and it shows up as unchanged over 30 days.
        if np.any(np.isnan(sequence)) or np.any(np.isnan(label)):
            continue
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

def prepare_ds(ds, batch_size):
    """Simple helper function to prepare a Tensorflow Dataset object."""
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)

def load_data(batch_size=128):
    """
    Load the data for each stock that data is available. Apply simple preprocessing,
    batch into sequences of 30 day history, split into train and validation
    sets, and then create Tensorflow Dataset objects for training.
    """
    data_folder = 'data'
    csv_files = sorted([str(f) for f in Path(data_folder).glob('*.csv')])[:10]
    # Load sequences for each stock
    all_sequences = []
    all_labels = []
    print('Loading Data...')
    for csvfile in tqdm(csv_files):
        # If this is a validation only stock, skip it
        if Path(csvfile).stem in VALIDATION_STOCKS:
            continue
        sequences, labels = load_and_preprocess(csvfile)
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    all_sequences = np.stack(all_sequences, axis=0)
    all_labels = np.array(all_labels)
    # Do an initial proper shuffle of the elements
    indices = np.arange(all_sequences.shape[0])
    np.random.shuffle(indices)
    all_sequences = all_sequences[indices]
    all_labels = all_labels[indices]
    # Turn it into a Tensorflow dataset
    seq_ds = Dataset.from_tensor_slices(all_sequences)
    label_ds = Dataset.from_tensor_slices(all_labels)
    ds = Dataset.zip((seq_ds, label_ds))
    ds = ds.shuffle(5000, reshuffle_each_iteration=True)
    num_val = int(round(0.2 * all_sequences.shape[0]))
    val_ds = ds.take(num_val)
    train_ds = ds.skip(num_val)
    # Batch and prefetch
    train_ds = prepare_ds(ds, batch_size)
    val_ds = prepare_ds(ds, batch_size)
    return train_ds, val_ds

def trade_evaluation(model, data, supervise_every=1):
    """
    An evaluation function to see how network predictions stack up over time.
    
    Parameters
    ----------
    data : ndarray
        ndarray of shape (60, 2). 60 days worth of data for one stock.
    """
    # First, preprocess the data
    initial_sequence = data[:30]
    s_min = initial_sequence.min(axis=0)
    s_max = initial_sequence.max(axis=0)
    data = (data - s_min[np.newaxis,:]) / (s_max[np.newaxis,:] - s_min[np.newaxis,:])
    sequence = data[:30]
    # Iteratively make predictions
    actual_prices = data[:,0]
    predicted_prices = data[:30,0].tolist()
    for i in range(30):
        predicted_change = float(model.predict(sequence[np.newaxis, :, :], verbose=0)[0])
        predicted_price = sequence[-1,0] + predicted_change
        predicted_prices.append(predicted_price)
        if (i + 1) % supervise_every == 0:
            # Use true price and volume for next part of sequence
            sequence = data[i + 1: i + 31]
        else:
            new_data = np.array([[predicted_price, sequence[-1, 1]]])
            sequence = np.concatenate([sequence[1:], new_data], axis=0)
    return np.array(predicted_prices), actual_prices

class TradeEvaluationCallback(tf.keras.callbacks.Callback):
    """A custom callback for testing the network as a trader"""
    def __init__(self, out_folder, supervise_every=10, **kwargs):
        super().__init__(**kwargs)

        # Make sure the folder exists
        Path(out_folder).mkdir(exist_ok=True)
        self.out_folder = out_folder

    def on_epoch_end(self, epoch, logs=None):
        # Run the evaluation on each stock
        for ticker in VALIDATION_STOCKS:
            data = pd.read_csv(str(Path('data', f'{ticker}.csv')), usecols=['Adj Close', 'Volume'])
            data = data.to_numpy()[-60:]
            for supervise_interval in [2, 10, 30]:
                predicted, true = trade_evaluation(self.model, data, supervise_interval)
                # Plot it
                saveas = str(Path(self.out_folder, f'{ticker}_{supervise_interval:02}.png'))
                fig, ax = plt.subplots()
                ax.plot(predicted, color='blue', linestyle='--', label='Predicted close')
                ax.plot(true, color='black', label='True close')
                ax.set_title(f'Network predictions for {ticker}\nSupervision every {supervise_interval} days')
                plt.savefig(saveas)
                plt.close()
    

class CustomLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sigmoid = layers.Activation('sigmoid')
    def call(self, x):
        x = self.sigmoid(x)
        return (2 * x) - 0.5

def make_model():
    """Simple model which wil predict next day's returns"""
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'))
    model.add(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'))
    model.add(layers.LSTM(128, return_sequences=False, kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='linear'))
    # model.add(CustomLayer())
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
        # metrics=[
        #     DirectionalAccuracy()
        # ],
        run_eagerly=True
    )
    return model

def train():
    """Main training function for LSTM"""
    # Load up the data
    train_ds, val_ds = load_data()
    # Build thfe model
    model = make_model()
    # Setup some callbacks
    logger = tf.keras.callbacks.CSVLogger('lstm_log.csv')
    ckpt = tf.keras.callbacks.ModelCheckpoint('trained_lstm', save_best_only=True)
    trade_eval = TradeEvaluationCallback('lstm_trading_evaluation')
    # Run training
    model.fit(train_ds, validation_data=val_ds, epochs=20, 
              callbacks=[logger, ckpt, trade_eval])

if __name__ == '__main__':
    train()
