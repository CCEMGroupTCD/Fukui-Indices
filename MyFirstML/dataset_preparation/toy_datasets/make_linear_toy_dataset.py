"""
This script makes a linear toy dataset for testing purposes.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from MyFirstML.utils.input_output import write_to_csv

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'regression', 'linear_toy_dataset.csv')
    n_samples = 100
    comment = 'Linear toy dataset.'

    x = np.linspace(0, 1, n_samples)
    y = x + 1
    data = pd.DataFrame({'x1': x, 'y': y})

    write_to_csv(df=data, output_path=outpath, comment=comment, index=False)


