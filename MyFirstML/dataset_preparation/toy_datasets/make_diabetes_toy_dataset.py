"""
This script makes a linear toy dataset for testing purposes.
"""
from pathlib import Path
from sklearn.datasets import load_diabetes

from MyFirstML.utils.input_output import write_to_csv

if __name__ == '__main__':
    outpath = Path('..', '..', '..', 'data', 'toy_datasets', 'regression', 'diabetes_toy_dataset.csv')
    comment = 'Diabetes toy dataset.'

    data = load_diabetes(as_frame=True).frame

    write_to_csv(df=data, output_path=outpath, comment=comment, index=False)


