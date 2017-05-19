from glob import glob

import numpy as np
import os
from tqdm import tqdm


def _read_contents_of_dir(dir, skiprows=0, sequence=False):
    array = []
    label = os.path.split(dir)[-1]
    for fn in tqdm(glob(os.path.join(dir, '*.csv')), desc='Reading from ' + label):
        array.append(np.loadtxt(fn, delimiter=',', skiprows=skiprows))
    return np.array(array) if sequence else np.vstack(array)


def read_data_from_disk(input_dir, target_dirs=None, start=None, end=None):
    if 'sequence' in input_dir:
        X = _read_contents_of_dir(input_dir, skiprows=1, sequence=True)
    else:
        X = _read_contents_of_dir(input_dir)

    Y = []
    if type(target_dirs) is str:
        Y = [ _read_contents_of_dir(target_dirs) ]
    elif target_dirs is not None:
        Y = [ _read_contents_of_dir(dir) for dir in target_dirs ]
    Y = [ y.ravel() if y.shape[1] == 1 else y for y in Y ]

    if start and end:
        X = X[start:end]
        Y = [ y[start:end] for y in Y ]
    elif start:
        X = X[start:]
        Y = [ y[start:] for y in Y ]
    elif end:
        X = X[:end]
        Y = [ y[:end] for y in Y ]

    return X, Y

if __name__ == '__main__':
    X, Y = read_data_from_disk('../src/main/resources/physionet2012/sequence',
                                [ '../src/main/resources/physionet2012/mortality',
                                  '../src/main/resources/physionet2012/los_bucket' ],
                                start=0, end=1234)
    print(X[0])
    for y in Y:
        print(y[0])
    print(X.shape, len(Y))
