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

import utils
import lstm_eval as eval

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

def load_and_preprocess(csvfile, target_days_out=1, use_volume=True, 
                        discretize_target=True, smooth_data=True):
    """
    Load and preprocess the data for one stock.
    
    Parameters
    ----------
    csvfile : str
        The path to the file where data is loaded from.
    target_days_out : int
        The number of days in the future to set the target variable (what the
        network is trying to predict).
    use_volume : bool
        Whether or not volume should be used as a feature.
    discretize_target : bool
        If True, turn target variable into discretized variable as follows:
            0: y < -0.03 (loss of greater than 3%)
            1: -0.03 < y < -0.01
            2: -0.01 < y < 0.01
            3: 0.01 < y < 0.03
            4: 0.03 < y
    smooth_data : bool
        If True, turn very noisy stock data into more smooth exponential
        moving average.
    """
    cols = ['Adj Close', 'Volume'] if use_volume else ['Adj Close']
    data = pd.read_csv(csvfile, usecols=cols)
    # Break it up into sequences of 30 days worth of data
    sequences = []
    labels = []
    for i in range(0, data.shape[0] - 29 - target_days_out, 30):
        sequence = data.iloc[i: i + 30].to_numpy()

        # Optionally smooth the data
        if smooth_data:
            ema = sequence[0]
            emas = [ema]
            for item in sequence[1:]:
                ema = (ema / 2) + (item / 2)
                emas.append(ema)
            sequence = np.stack(emas, axis=0)

        # Normalize so that all prices are indexed to the last day in the sequence
        norm = sequence[-1]
        sequence = sequence / norm[np.newaxis,:]
        # # Normalize with min/max scaling
        # s_min = sequence.min(axis=0)
        # s_max = sequence.max(axis=0)
        # sequence = (sequence - s_min[np.newaxis,:]) / (s_max[np.newaxis,:] - s_min[np.newaxis,:])
        # Get label and make sure it is also normalized
        tgt_price = data.loc[i + 29 + target_days_out, 'Adj Close'] / norm[0]
        # tgt_price = (data.loc[i + 29 + target_days_out, 'Adj Close'] - s_min[0]) / (s_max[0] - s_min[0])
        label = tgt_price - sequence[-1, 0]
        # If any are NaN, don't keep it. This occurs when there is an error in
        # reporting the stock price and it shows up as unchanged over 30 days.
        if np.any(np.isnan(sequence)) or np.any(np.isnan(label)):
            continue
        # Optionally turn it into a discrete label instead of continuous
        if discretize_target:
            tmp = (label + 0.03) / 0.02
            tmp = min(max(tmp, 0), 4)
            label = int(np.ceil(tmp))
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

def prepare_ds(ds, batch_size):
    """Simple helper function to prepare a Tensorflow Dataset object."""
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)

def load_data(batch_size=128, use_volume=True, target_days_out=1, 
              discretize_target=False, smooth_data=False, debug=False):
    """
    Load the data for each stock that data is available. Apply simple preprocessing,
    batch into sequences of 30 day history, split into train and validation
    sets, and then create Tensorflow Dataset objects for training.
    """
    data_folder = 'data'
    csv_files = sorted([str(f) for f in Path(data_folder).glob('*.csv')])
    if debug:
        csv_files = csv_files[:10]
    # Load sequences for each stock
    all_sequences = []
    all_labels = []
    holdout_sequences = []
    holdout_labels = []
    print('Loading Data...')
    for csvfile in tqdm(csv_files):
        # Check if this is part of holdout set
        holdout = Path(csvfile).stem in VALIDATION_STOCKS
        local_discretize = (not holdout) and discretize_target
        sequences, labels = load_and_preprocess(csvfile, target_days_out, 
                                                use_volume, local_discretize,
                                                smooth_data)
        # If this is a validation only stock, add it to holdout data
        if holdout:
            holdout_sequences.extend(sequences)
            holdout_labels.extend(labels)
        else:
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

    # Prepare holdout dataset too
    holdout_seq_ds = Dataset.from_tensor_slices(holdout_sequences)
    holdout_label_ds = Dataset.from_tensor_slices(holdout_labels)
    holdout_ds = Dataset.zip((holdout_seq_ds, holdout_label_ds))
    holdout_ds = prepare_ds(holdout_ds, batch_size)

    return train_ds, val_ds, holdout_ds







def make_model(discretize=False):
    """Simple model which wil predict next day's returns"""
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'))
    model.add(layers.LSTM(128, return_sequences=True, kernel_initializer='he_normal'))
    model.add(layers.LSTM(128, return_sequences=False, kernel_initializer='he_normal'))
    # Final layer will depend on continuous vs discrete output variable
    if discretize:
        final_layer = layers.Dense(5, activation='softmax')
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        final_layer = layers.Dense(1, activation='linear')
        loss = tf.keras.losses.MeanSquaredError()
    model.add(final_layer)
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4),
        # metrics=[
        #     DirectionalAccuracy()
        # ],
        run_eagerly=True
    )
    return model

def plot_label_dist(ds):
    """Plot a histogram of all labels"""
    all_labels = []
    count = 0
    for _, labels in ds:
        all_labels.extend(labels.numpy().tolist())
        count += 1
        if count >= 100:
            break
    fig, ax = plt.subplots()
    bins = np.linspace(-0.5, 4.5, 6)
    ax.hist(all_labels, color='dodgerblue', alpha=0.4, bins=bins)
    plt.show()

def train(output_dir, use_volume=False, discretize=False, target_days_out=1,
          smooth_data=False, debug=False):
    """Main training function for LSTM"""
    # Make an output directory
    output_dir = utils.setup_output_dir(output_dir)

    # Load up the data
    train_ds, val_ds = load_data(use_volume=use_volume, 
                                 target_days_out=target_days_out,
                                 discretize_target=discretize,
                                 smooth_data=smooth_data,
                                 debug=debug)
    
    plot_label_dist(train_ds)
    stop

    # Build thfe model
    model = make_model(discretize=discretize)
    # Setup some callbacks
    logger = tf.keras.callbacks.CSVLogger(str(Path(output_dir, 'lstm_log.csv')))
    ckpt = tf.keras.callbacks.ModelCheckpoint(str(Path(output_dir, 'trained_lstm')), 
                                              save_best_only=True)
    callbacks = [logger, ckpt]
    if not discretize:
        trade_eval = eval.AutoregressiveEvalCallback(str(Path(output_dir, 'lstm_trading_evaluation')), 
                                                     use_volume=use_volume)
        callbacks.append(trade_eval)
    # Run training
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

if __name__ == '__main__':
    # Output directory
    output_dir = 'LSTM_OUTPUT'

    # Set training parameters
    use_volume = False
    discretize = True
    target_days_out = 5
    smooth_data = True
    debug = True

    train(output_dir, use_volume, discretize, target_days_out, 
          smooth_data=smooth_data, debug=debug)
