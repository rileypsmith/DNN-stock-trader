"""
A simple LSTM to learn from historical stock data and decide whether to
buy, sell, or hold for a stock daily over a period of time.

@author: Riley Smith
Created: 08/30/2023
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers, Sequential

warnings.simplefilter('ignore')

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
    data = pd.read_csv(csvfile)[['Adj Close', 'Volume']]
    # Normalize volume with mean/std normalization
    data['Volume']  = (data.Volume - data.Volume.mean()) / data.Volume.std()
    # Break it up into sequences of 30 days worth of data
    sequences = []
    labels = []
    for i in range(0, data.shape[0] - 30, 30):
        sequence = data.iloc[i: i + 30]
        # Normalize closing price with min/max scaling
        s_min = sequence['Adj Close'].min()
        s_max = sequence['Adj Close'].max()
        sequence['Adj Close'] = (sequence['Adj Close'] - s_min) / (s_max - s_min)
        sequence = sequence.to_numpy()
        tgt_price = (data.loc[i + 30, 'Adj Close'] - s_min) / (s_max - s_min)
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
    csv_files = sorted([str(f) for f in Path(data_folder).glob('*.csv')])
    # Load sequences for each stock
    all_sequences = []
    all_labels = []
    print('Loading Data...')
    for csvfile in tqdm(csv_files):
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
    # Run training
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[logger, ckpt])

if __name__ == '__main__':
    train()
