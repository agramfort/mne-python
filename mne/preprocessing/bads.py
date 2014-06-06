# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Jacobo Sitt <jdsitt@gmail.com>
# License: BSD (3-clause)


import numpy as np
from scipy.stats import zscore

from ..utils import logger


def find_outlier_adaptive(X, threshold=3, max_iter=10):
    """Find outliers based on iterative z-scoring

    Parameters
    ----------
    X : np.ndarray of float
        The metric under question.
    threshold : int | float
        The value above which a feature is classified as outlier.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n ica components)
        The outlier indices.
    """
    this_x = np.abs(X.copy())
    my_mask = np.zeros(len(this_x), dtype=np.bool)
    i_iter = 0
    msg = 'iteration %i : total outliers: %i'

    while i_iter < max_iter:
        this_x = np.ma.masked_array(this_x, my_mask)
        this_z = zscore(this_x)
        local_bad = np.abs(this_z) > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        logger.info(msg % (i_iter, sum(my_mask)))
        if not np.any(local_bad):
            logger.info('converged.')
            break
        i_iter += 1

    bad_idx = np.nonzero(this_x.mask)[0]
    return bad_idx
