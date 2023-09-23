"""
Evaluation functions for the LSTM.

@author: Riley Smith
Created: 9/17/2023
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

def preprocess_sequence(seq):
    """Preprocess a sequence of data"""
    # First, preprocess the data
    initial_sequence = seq[:30]
    norm = initial_sequence[-1]
    seq = seq / norm[np.newaxis, :]
    return seq

def autoregressive_eval(model, data, supervise_every=1, use_volume=True):
    """
    An evaluation function to see how network predictions stack up over time.
    This one is specifically for the autoregressive model looking one day in the
    future.
    
    Parameters
    ----------
    model : tf.keras.Model
        The trained network being evaluated
    data : ndarray
        ndarray of shape (60, 2). 60 days worth of data for one stock.
    supervise_every : int
        Number of days to run auto-regressive predictions before correcting
        with true price.
    use_volume : bool
        Whether or not volume is a feature for predictions.
    """
    data = preprocess_sequence(data)
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
            if use_volume:
                new_data = np.array([[predicted_price, sequence[-1, 1]]])
                sequence = np.concatenate([sequence[1:], new_data], axis=0)
            else:
                sequence = np.array(sequence[1:].ravel().tolist() + [predicted_price])[:,np.newaxis]
    return np.array(predicted_prices), actual_prices

class AutoregressiveEvalCallback(tf.keras.callbacks.Callback):
    """A custom callback for testing the network as a trader"""
    def __init__(self, out_folder, use_volume=True, **kwargs):
        super().__init__(**kwargs)

        # Make sure the folder exists
        Path(out_folder).mkdir(exist_ok=True)
        self.out_folder = out_folder
        self.use_volume = use_volume

    def on_epoch_end(self, epoch, logs=None):
        # Run the evaluation on each stock
        for ticker in VALIDATION_STOCKS:
            cols = ['Adj Close', 'Volume'] if self.use_volume else ['Adj Close']
            data = pd.read_csv(str(Path('data', f'{ticker}.csv')), usecols=cols)
            data = data.to_numpy()[-60:]
            for supervise_interval in [1, 2, 10, 30]:
                predicted, true = autoregressive_eval(self.model, data, 
                                                      supervise_interval, self.use_volume)
                # Plot it
                saveas = str(Path(self.out_folder, f'EPOCH{epoch:03}_{ticker}_{supervise_interval:02}.png'))
                fig, ax = plt.subplots()
                ax.plot(predicted, color='blue', linestyle='--', label='Predicted close')
                ax.plot(true, color='black', label='True close')
                ax.set_title(f'Network predictions for {ticker}\nSupervision every {supervise_interval} days')
                ax.legend(loc='lower right')
                plt.savefig(saveas)
                plt.close()

def categorical_eval(model, ds):
    """
    An evaluation function for the categorical stock prediction problem.
    
    Take a sequence of data, predict the class (category corresponding to stock
    movement), and also compute the true label.

    Parameters
    ----------
    model : tf.keras.Model
        The neural network being evaluated.
    ds : tf.data.Dataset
        Tensorflow Dataset object serving up preprocessed batches and
        corresponding labels.
    """
    all_preds = []
    all_labels = []
    for batch, label in ds:
        # Get predictions for this batch
        preds = model.predict(batch).argmax(axis=1)
        all_preds.append(preds)
        all_labels.append(label.numpy())
    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    # Aggregate results by class
    aggregate_results = {}
    for class_idx in range(5):
        returns = all_labels[np.where(all_preds == class_idx)]
        if returns.size == 0:
            d = {'av': 'NA', 'std': 'NA'}
        else:
            d = {
                'av': returns.mean(),
                'std': returns.std()
            }
        aggregate_results[class_idx] = d
    print('\n\n\nAGGREGATE RESULTS: ', aggregate_results, '\n\n\n')
    return aggregate_results

class CategoricalEvalCallback(tf.keras.callbacks.Callback):
    """Custom callback for evaluating performance of categorical model"""
    def __init__(self, ds, outfile, **kwargs):
        """
        Parameters
        ----------
        ds : tf.data.Dataset
            The Tensorflow Dataset object on which to run evaluation.
        """
        super().__init__(**kwargs)
        self.ds = ds
        # Setup output file
        self.outfile = outfile
        with open(self.outfile, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = ['Epoch']
            for class_idx in range(5):
                row.append(f'CLASS{class_idx}_mean')
                row.append(f'CLASS{class_idx}_std')
            writer.writerow(row)

    def on_epoch_end(self, epoch, logs=None):
        """Run evaluation once every epoch"""
        results = categorical_eval(self.model, self.ds)
        # Write it to CSV
        row = [epoch]
        for class_dict in results.values():
            row.append(class_dict['av'])
            row.append(class_dict['std'])
        with open(self.outfile, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        

