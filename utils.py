"""
Utility functions for DNN stock trading.

@author: Riley Smith
Created: 9/17/2023
"""

from pathlib import Path

def setup_output_dir(output_dir):
    """
    Setup an output directory. If the given directory already exists, append
    numbers to the end to create a unique name and then create it.
    
    Parameters
    ----------
    output_dir : str
        The path to the directory you want to create.
    
    Returns
    -------
    str
        The Path to the directory that was created.
    """
    appended = False
    tries = 0
    while Path(output_dir).exists():
        if not appended:
            output_dir = str(Path(output_dir).absolute()) + f'_{tries:03}'
        else:
            output_dir = str(Path(output_dir).absolute())[:-4] + f'_{tries:03}'
        tries += 1
        appended = True
    Path(output_dir).mkdir()
    return str(output_dir)