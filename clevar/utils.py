import numpy as np

veclen = np.vectorize(len)
def none_val(value, none_value):
    """
    Set default value to be returned if input is None

    Parameters
    ----------
    value:
        Input value
    none_value:
        Value to be asserted if input is None

    Returns
    -------
    type(value), type(none_value)
        Value if value is not None else none_value
    """
    return value if value is not None else none_value
def logbins(values, nbins):
    """
    Make log10 spaced bins from data

    Parameters
    ----------
    values: array
        Data
    nbins: int
        Number of bins

    Returns
    -------
    ndarray
        Log10 spaced bins based on values
    """
    logvals = np.log10(values)
    return np.logspace(logvals.min(), logvals.max(), nbins)
