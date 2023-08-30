"""
A simple LSTM to learn from historical stock data and decide whether to
buy, sell, or hold for a stock daily over a period of time.

@author: Riley Smith
Created: 08/30/2023
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import layers, Sequential

class DirectionalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.correct_count = 0
        self.total_count = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        comparison = tf.equal(tf.math.sign(y_true), tf.math.sign(y_pred))
        self.correct_count += tf.math.reduce_sum(comparison)
        self.total_count += tf.size(y_true)

    def result(self):
        return self.correct_count / self.total_count
    
    def reset_states(self):
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
        sequence = data.iloc[i: i + 30].to_numpy()
        label = data.loc[i + 30, 'Adj Close'] - data.loc[i + 29, 'Adj Close']
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

def load_data(batch_size=128):
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
    # Turn it into a Tensorflow dataset
    seq_ds = Dataset.from_tensor_slices(all_sequences)
    label_ds = Dataset.from_tensor_slices(all_labels)
    ds = Dataset.zip((seq_ds, label_ds))
    ds = ds.shuffle(5000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def make_model():
    """Simple model which wil predict next day's returns"""
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=False))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            DirectionalAccuracy()
        ],
        callbacks=[
            tf.keras.callbacks.CSVLogger('lstm_log.csv')
        ]
    )
    return model

def train():
    """Main training function for LSTM"""
    # Load up the data
    ds = load_data()
    # Build the model
    model = make_model()
    # Run training
    model.fit(ds, epochs=20)

if __name__ == '__main__':
    train()
