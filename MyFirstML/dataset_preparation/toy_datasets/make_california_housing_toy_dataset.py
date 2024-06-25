"""
This script makes a linear toy dataset for testing purposes.
"""
from pathlib import Path
from sklearn.datasets import fetch_california_housing

from MyFirstML.utils.input_output import write_to_csv

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'regression', 'california_housing_toy_dataset.csv')
    comment = 'California housing toy dataset.'

    data = fetch_california_housing(as_frame=True).frame

    write_to_csv(df=data, output_path=outpath, comment=comment, index=False)

